import loggoing  
import maths     
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm.auto import tqdm
import numpy as np
import random
from torch.optim.lr_scheduler import LambdaLR
from typing import Optional, List
import os

try:
    import jiwer
except Exception:
    jiwer = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# --- Pose Encoder Modifications ---

class JointAttentionSelector(nn.Module):
    def __init__(self, num_joints=25, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(num_joints * 4, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_joints)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, pose_tensor):
        B, T, K, C = pose_tensor.shape
        pooled = pose_tensor.mean(dim=1)
        flat = pooled.reshape(B, -1)
        attn = self.fc2(torch.relu(self.fc1(flat)))
        attn = self.softmax(attn)
        pose_tensor = pose_tensor * attn[:, None, :, None]
        return pose_tensor, attn

def normalize_adjacency(A: torch.Tensor) -> torch.Tensor:
    A = A.clone().float()
    A = A + torch.eye(A.shape[0], device=A.device)
    deg = A.sum(dim=1)
    deg_inv_sqrt = torch.pow(deg, -0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0
    D_inv_sqrt = torch.diag(deg_inv_sqrt)
    return D_inv_sqrt @ A @ D_inv_sqrt

def build_adjacency(num_joints: int, layout: str = 'auto', partitions: int = 3, center: Optional[int] = None) -> torch.Tensor:
    skeletons = {
        25: [(0,1),(1,2),(2,3),(3,4),(1,5),(5,6),(6,7),(1,8),(8,9),(9,10),(10,11),
             (8,12),(12,13),(13,14),(0,15),(15,17),(0,16),(16,18),(14,19),(19,20),(14,21),(11,22),(22,23),(11,24)],
        17: [(0,1),(1,2),(2,3),(0,4),(4,5),(5,6),(0,7),(7,8),(8,9),(7,10),(10,11),(11,12),(7,13),(13,14),(14,15),(0,16)],
        21: [(i, i+1) for i in range(20)]
    }
    if layout == 'auto':
        edges = skeletons.get(num_joints, [(i, i+1) for i in range(num_joints-1)])
    else:
        edges = [(i, i+1) for i in range(num_joints-1)]
    A = torch.zeros((num_joints, num_joints), dtype=torch.float32)
    for (u,v) in edges:
        if u < num_joints and v < num_joints:
            A[u, v] = 1.0
            A[v, u] = 1.0
    A_norm = normalize_adjacency(A)
    if partitions == 1:
        return A_norm.unsqueeze(0)
    if center is None:
        center = num_joints // 2
    A_parts = torch.zeros((partitions, num_joints, num_joints), dtype=torch.float32)
    A_parts[0] = torch.eye(num_joints)
    mask_center = torch.zeros((num_joints, num_joints), dtype=torch.float32)
    mask_center[center, :] = A[center, :]
    mask_center[:, center] = A[:, center]
    A_parts[1] = normalize_adjacency(mask_center)
    remaining = A_norm - A_parts[1]
    remaining[remaining < 0] = 0.0
    A_parts[2] = remaining
    return A_parts

class STGCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, A_parts: torch.Tensor, stride=1):
        super().__init__()
        P, V, _ = A_parts.shape
        self.register_buffer('A_init', A_parts)
        self.edge_importance = nn.Parameter(torch.ones(P, dtype=torch.float32))
        self.gcn = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        self.tcn = nn.Conv2d(out_channels, out_channels, kernel_size=(9, 1), stride=(stride, 1), padding=(4, 0))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)) if (in_channels != out_channels or stride != 1) else nn.Identity()

    def forward(self, x):
        N, C, T, V = x.shape
        P = self.A_init.shape[0]
        agg = torch.zeros((N, C, T, V), device=x.device, dtype=x.dtype)
        for p in range(P):
            Ap = self.A_init[p] * self.edge_importance[p]
            agg += torch.einsum('nctv,vw->nctw', x, Ap.to(x.device))
        x_gcn = self.gcn(agg)
        x_tcn = self.tcn(x_gcn)
        x_tcn = self.bn(x_tcn)
        res = self.residual(x)
        return self.relu(x_tcn + res)

class ProPoseEncoder(nn.Module):
    def __init__(self, in_channels=4, num_joints=25, hidden=128, out_dim=256, num_layers=3, distill_teacher=None):
        super().__init__()
        self.joint_selector = JointAttentionSelector(num_joints=num_joints, hidden_dim=hidden)
        A_parts = build_adjacency(num_joints, layout='auto', partitions=3)
        self.stgcn_layers = nn.ModuleList()
        self.stgcn_layers.append(STGCNLayer(in_channels*4, hidden, A_parts))
        for _ in range(num_layers-2):
            self.stgcn_layers.append(STGCNLayer(hidden, hidden, A_parts))
        self.stgcn_layers.append(STGCNLayer(hidden, out_dim, A_parts))
        self.distill_teacher = distill_teacher

    def forward(self, pose_tensor):
        pos = pose_tensor
        vel = torch.zeros_like(pos)
        acc = torch.zeros_like(pos)
        jerk = torch.zeros_like(pos)
        vel[:, 1:] = pos[:, 1:] - pos[:, :-1]
        acc[:, 1:] = vel[:, 1:] - vel[:, :-1]
        jerk[:, 1:] = acc[:, 1:] - acc[:, :-1]
        feats = torch.cat([pos, vel, acc, jerk], dim=-1)
        feats, attn = self.joint_selector(feats)
        feats = feats.permute(0, 3, 1, 2).contiguous()
        for layer in self.stgcn_layers:
            feats = layer(feats)
        per_frame = feats
        pooled = feats.mean(dim=2).mean(dim=-1)
        seq_emb = per_frame.mean(dim=-1).permute(0,2,1).contiguous()
        if self.distill_teacher is not None and self.training:
            with torch.no_grad():
                teacher_out = self.distill_teacher(pose_tensor)
            return pooled, seq_emb, teacher_out
        return pooled, seq_emb

# A simplified 1D temporal convolution block
class TemporalConv1DBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=9, stride=1, dilation=1):
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation
        self.conv = nn.Conv1d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_c)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


# TwoStreamEncoder and KeypointSelector removed. Use ProPoseEncoder instead.

# --- Fusion Module Modifications ---

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.relu(self.bn(x))

class MultiCoAttentionFusion(nn.Module):
    def __init__(self, video_dim, pose_dim, d_model=768, num_layers=8, nhead=12, dropout=0.1):
        super().__init__()
        # Using depthwise separable convs for initial pose projection
        self.pose_proj = DepthwiseSeparableConv(pose_dim, d_model, kernel_size=3, padding=1)
        # video projection
        self.video_proj = nn.Linear(video_dim, d_model)
        # cross-modal encoder (transformer) to fuse sequence-level tokens
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=min(num_layers, 6))
        self.adaptive_pool = nn.AdaptiveAvgPool1d(16)
        # final projection back to decoder d_model
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, v_feats: Optional[torch.Tensor], p_feats: torch.Tensor):
        """
        v_feats: Optional[Tensor] (B, T_v, video_dim) or None
        p_feats: Tensor either (B, T_p, C) or (B, C) if pooled. We support both.
        Returns fused features shaped (B, T_f, d_model) suitable for encoder_outputs.last_hidden_state
        """
        B = p_feats.shape[0]
        # handle pooled pose embeddings
        if p_feats.ndim == 2:
            # expand to temporal sequence by repeating
            p_seq = p_feats.unsqueeze(1).repeat(1, self.adaptive_pool.output_size[0], 1)  # (B, T_pool, C)
        else:
            p_seq = p_feats

        # project pose features
        # transformer expects (S, N, E)
        p_proj = self.pose_proj(p_seq.permute(0,2,1)).permute(2,0,1)  # (T_p, B, d_model)

        if v_feats is not None:
            v_proj = self.video_proj(v_feats)  # (B, T_v, d_model)
            v_proj = v_proj.permute(1,0,2)  # (T_v, B, d_model)
            # concatenate along sequence dimension
            seq = torch.cat([p_proj, v_proj], dim=0)  # (T_p+T_v, B, d_model)
        else:
            seq = p_proj

        fused = self.transformer(seq)  # (S, B, d_model)
        fused = fused.permute(1,0,2).contiguous()  # (B, S, d_model)
        fused = self.out_proj(fused)
        return fused


# --- Dataset and Trainer Modifications ---

def sample_frame_indices(total_frames: int, num_frames: int, stride: int = 1, is_train: bool = True):
    if total_frames <= 0:
        return np.zeros(num_frames, dtype=int)
    effective_length = (num_frames - 1) * stride + 1
    if total_frames <= effective_length:
        return np.linspace(0, max(total_frames - 1, 0), num_frames, dtype=int)
    if is_train:
        start = np.random.randint(0, total_frames - effective_length + 1)
    else:
        start = (total_frames - effective_length) // 2
    return np.arange(start, start + effective_length, stride)[:num_frames]

class SignDataset(Dataset):
    def __init__(self, df: pd.DataFrame, data_dir: str, tokenizer_name: str = "google/flan-t5-small",
                 num_frames: int = 16, temporal_stride: int = 2,
                 max_text_len: int = 64, is_train: bool = True, modality_dropout: float = 0.15,
                 curriculum_bucket_fn=None):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.num_frames = num_frames
        self.temporal_stride = temporal_stride
        self.is_train = is_train
        self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)
        self.modality_dropout = modality_dropout
        self.max_text_len = max_text_len
        self.curriculum_bucket_fn = curriculum_bucket_fn
        valid = []
        for _, r in self.df.iterrows():
            uid = r['uid']
            p = f"{self.data_dir}/{uid}.npy"
            if os.path.exists(p):
                valid.append(r)
        self.df = pd.DataFrame(valid).reset_index(drop=True)
        if len(self.df) == 0:
            raise RuntimeError("No valid samples found in data_dir with .npy files")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        uid = row['uid']
        pose_path = f"{self.data_dir}/{uid}.npy"
        pose_tensor = self._load_pose_npy(pose_path)
        # normalize and apply confidence weighting
        pose_tensor = self._normalize_pose(pose_tensor)

        # Apply attention-based keyframe selection if available
        try:
            pose_tensor = self._select_keyframes(pose_tensor, self.num_frames, self.temporal_stride)
        except Exception:
            pose_tensor = self._select_keyframes_simple(pose_tensor, self.num_frames, self.temporal_stride)

        if self.is_train and random.random() < self.modality_dropout:
            pose_tensor = torch.zeros_like(pose_tensor)
        else:
            # Apply pose augmentations during training
            if self.is_train:
                pose_tensor = self._augment_pose(pose_tensor)
        text = row.get('text', '')
        enc = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_text_len, return_tensors='pt')
        input_ids = enc['input_ids'].squeeze(0)
        attention_mask = enc['attention_mask'].squeeze(0)
        # compute a difficulty metric for curriculum learning (mean motion magnitude)
        difficulty = float(torch.mean(torch.norm(pose_tensor[:, :, :3], dim=-1)).item())
        return {'pose': pose_tensor, 'input_ids': input_ids, 'attention_mask': attention_mask, 'text': text, 'uid': uid, 'difficulty': difficulty}

    def _load_pose_npy(self, path):
        try:
            pose_data = np.load(path)
            return torch.tensor(pose_data, dtype=torch.float32)
        except Exception as e:
            logger.error(f"Failed to load .npy file from {path}: {e}")
            return torch.zeros(self.num_frames, 576, 4)

    def _normalize_pose(self, pose_tensor):
        # pose_tensor: (T, K, 4) where last dim is x,y,z,confidence
        pose = pose_tensor.clone()
        if pose.ndim != 3 or pose.shape[-1] < 4:
            return pose
        coords = pose[..., :3]
        conf = pose[..., 3:4]
        # weight coordinates by confidence
        coords = coords * conf
        # zero-out very low confidence joints
        coords[conf < 0.01] = 0.0
        pose[..., :3] = coords
        return pose

    def _select_keyframes_simple(self, pose_tensor, num_frames, temporal_stride):
        # fallback simple sampling
        total_frames = pose_tensor.shape[0]
        indices = sample_frame_indices(total_frames, num_frames, temporal_stride, self.is_train)
        return pose_tensor[indices]

    def _select_keyframes(self, pose_tensor, num_frames, temporal_stride):
        # Attention-based keyframe selection (lightweight trainable MLP per dataset instance)
        # pose_tensor: (T, K, C)
        T, K, C = pose_tensor.shape
        # compute frame-level descriptor: mean joint motion magnitude + mean confidence
        coords = pose_tensor[..., :3]  # (T,K,3)
        conf = pose_tensor[..., 3]     # (T,K)
        motion = torch.zeros((T, K), dtype=torch.float32)
        motion[1:] = torch.norm(coords[1:] - coords[:-1], dim=-1)
        frame_desc = torch.cat([motion.mean(dim=1, keepdim=True), conf.mean(dim=1, keepdim=True)], dim=1)  # (T,2)
        # simple attention scorer (trainable)
        if not hasattr(self, '_keyframe_scorer'):
            self._keyframe_scorer = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 1))
        scores = self._keyframe_scorer(frame_desc.to(torch.float32)).squeeze(-1)  # (T,)
        # select top-k frames (if training) or top evenly-spaced by score (if eval)
        k = num_frames
        if T <= k:
            indices = torch.arange(T)
        else:
            if self.is_train:
                _, idx = torch.topk(scores, k=k)
                indices = torch.sort(idx)[0]
            else:
                # take soft attention-weighted center mass for deterministic selection
                probs = torch.softmax(scores, dim=0)
                cumulative = torch.cumsum(probs, dim=0)
                targets = torch.linspace(0, 1, steps=k)
                indices = torch.searchsorted(cumulative, targets)
        indices = torch.clamp(indices, 0, T-1)
        return pose_tensor[indices]

    def _augment_pose(self, pose_tensor):
        # Random rotation (around Z), scaling and jitter
        T, K, C = pose_tensor.shape
        coords = pose_tensor[..., :3]
        conf = pose_tensor[..., 3:4]
        # rotation in xy plane
        angle = (random.random() * 2 - 1) * 0.1  # +-0.1 rad
        cos, sin = math.cos(angle), math.sin(angle)
        R = torch.tensor([[cos, -sin, 0.0],[sin, cos, 0.0],[0.0, 0.0, 1.0]], dtype=torch.float32)
        coords = torch.matmul(coords, R.to(coords.device))
        # scaling
        scale = 1.0 + (random.random() * 2 - 1) * 0.05
        coords = coords * scale
        # jitter
        jitter = (torch.randn_like(coords) * 0.01)
        coords = coords + jitter
        pose_tensor[..., :3] = coords
        pose_tensor[..., 3:4] = conf
        return pose_tensor

    def _select_keyframes(self, pose_tensor, num_frames, temporal_stride):
        total_frames = pose_tensor.shape[0]
        indices = sample_frame_indices(total_frames, num_frames, temporal_stride, self.is_train)
        return pose_tensor[indices]

class Video2TextSOTA(nn.Module):
    def __init__(self, pose_encoder, fusion, decoder, config):
        super().__init__()
        self.pose_encoder = pose_encoder
        self.fusion = fusion
        self.decoder = decoder
        self.config = config
        d_model = config.get('d_model', 512) # Reduced d_model
        self.ctc_linear = nn.Linear(d_model, config.get('vocab_size', self.decoder.config.vocab_size))
        self.label_smoothing = config.get('label_smoothing', 0.1)
        if self.label_smoothing > 0:
            self.seq_loss_fn = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        else:
            self.seq_loss_fn = nn.CrossEntropyLoss()
        self.ctc_loss_fn = nn.CTCLoss(blank=config.get('blank_idx', 0), zero_infinity=True)

    def forward(self, labels=None, attention_mask=None, pose=None, label_lengths=None):
        B = pose.shape[0]
        device = pose.device
        enc_out = self.pose_encoder(pose.to(device))
        # enc_out may be (pooled, seq_emb) or (pooled, seq_emb, teacher_out)
        teacher_out = None
        if isinstance(enc_out, tuple) and len(enc_out) == 3:
            pooled, seq_emb, teacher_out = enc_out
        elif isinstance(enc_out, tuple) and len(enc_out) == 2:
            pooled, seq_emb = enc_out
        else:
            pooled = enc_out
            seq_emb = None
        fused = self.fusion(None, seq_emb if seq_emb is not None else pooled)
        outputs = {}
        if self.config.get('use_ctc', True) and labels is not None:
            logits_ctc = self.ctc_linear(fused)
            log_probs = nn.functional.log_softmax(logits_ctc, dim=-1).permute(1,0,2)
            input_lengths = torch.full((B,), fused.shape[1], dtype=torch.long)
            target_lengths = label_lengths if label_lengths is not None else torch.sum(labels!=self.decoder.config.pad_token_id, dim=1)
            try:
                ctc_loss = self.ctc_loss_fn(log_probs, labels, input_lengths, target_lengths)
            except Exception as e:
                logger.warning('CTC loss failed: %s', e)
                ctc_loss = torch.tensor(0.0, device=device)
            outputs['ctc_loss'] = ctc_loss
        class EncWrap:
            pass
        encwrap = EncWrap()
        encwrap.last_hidden_state = fused
        if labels is not None:
            decoder_outputs = self.decoder(encoder_outputs=encwrap, labels=labels, attention_mask=attention_mask)
            seq_loss = decoder_outputs.loss
            outputs['seq_loss'] = seq_loss
        else:
            outputs['encoder_outputs'] = encwrap
        if 'seq_loss' in outputs and 'ctc_loss' in outputs:
            total = outputs['seq_loss'] * self.config.get('seq_weight', 1.0) + outputs['ctc_loss'] * self.config.get('ctc_weight', 0.3)
            outputs['loss'] = total
        elif 'seq_loss' in outputs:
            outputs['loss'] = outputs['seq_loss']
        elif 'ctc_loss' in outputs:
            outputs['loss'] = outputs['ctc_loss']
        # include distillation targets if available
        if teacher_out is not None:
            outputs['teacher_repr'] = teacher_out
            outputs['student_repr'] = pooled
        return outputs

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._init_data()
        self._init_models()
        self.scaler = torch.cuda.amp.GradScaler(enabled=config.get('use_amp', True))
        self.writer = SummaryWriter(config.get('log_dir', 'logs'))
        self.global_step = 0
        # early stopping state
        self.best_val = float('inf')
        self.no_improve_epochs = 0
        self.early_stop_patience = self.config.get('early_stopping_patience', 5)

    def _init_data(self):
        train_df = self.config['train_df']
        val_df = self.config['val_df']
        self.train_ds = SignDataset(train_df, self.config['data_dir'], tokenizer_name=self.config['tokenizer_name'],
                                    num_frames=self.config['num_frames'],
                                    temporal_stride=self.config['temporal_stride'], is_train=True, modality_dropout=self.config.get('modality_dropout',0.15))
        self.val_ds = SignDataset(val_df, self.config['data_dir'], tokenizer_name=self.config['tokenizer_name'],
                                  num_frames=self.config['num_frames'],
                                  temporal_stride=self.config['temporal_stride'], is_train=False, modality_dropout=0.0)
        self.train_loader = DataLoader(self.train_ds, batch_size=self.config['batch_size'], shuffle=True, num_workers=self.config.get('num_workers',4), pin_memory=True)
        self.val_loader = DataLoader(self.val_ds, batch_size=self.config['batch_size'], shuffle=False, num_workers=self.config.get('num_workers',4), pin_memory=True)

    def _init_models(self):
        # Teacher for distillation (optional)
        teacher_encoder = None
        if self.config.get('teacher_encoder_path', None) is not None:
            try:
                te_path = self.config['teacher_encoder_path']
                # assume teacher is a checkpoint that can be loaded with torch.load
                ckpt = torch.load(te_path, map_location='cpu')
                # user must ensure checkpoint contains a 'state_dict' or a model object
                # For now, we leave it as None and expect user to provide a callable teacher via config
                teacher_encoder = ckpt.get('model', None) if isinstance(ckpt, dict) else None
            except Exception as e:
                logger.warning('Could not load teacher encoder checkpoint: %s', e)
        # Optionally load a larger T5 teacher for decoder distillation
        teacher_t5 = None
        if self.config.get('teacher_t5_name', None):
            try:
                teacher_t5 = T5ForConditionalGeneration.from_pretrained(self.config['teacher_t5_name']).to(self.device)
                teacher_t5.eval()
                for p in teacher_t5.parameters():
                    p.requires_grad = False
            except Exception as e:
                logger.warning('Failed to load teacher T5: %s', e)
        pose_enc = ProPoseEncoder(
            in_channels=4,
            num_joints=576,
            hidden=self.config.get('pose_hidden',128),
            out_dim=self.config.get('pose_out',256),
            num_layers=3,
            distill_teacher=teacher_encoder
        )
        fusion = MultiCoAttentionFusion(
            video_dim=self.config.get('video_feat_dim',512),
            pose_dim=self.config.get('pose_out',256),
            d_model=self.config.get('d_model',512)
        )
        decoder = T5ForConditionalGeneration.from_pretrained(self.config['tokenizer_name'])
        self.model = Video2TextSOTA(pose_enc, fusion, decoder, self.config).to(self.device)
        # attach a teacher model reference for training distillation losses
        self.teacher_t5 = teacher_t5
        no_decay = ["bias", "LayerNorm.weight"]
        grouped = [
            {"params": [p for n,p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": self.config.get('weight_decay',1e-2)},
            {"params": [p for n,p in self.model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
        ]
        self.optimizer = optim.AdamW(grouped, lr=self.config.get('lr',5e-5), betas=(0.9,0.999), eps=1e-8)
        total_steps = max(1, len(self.train_loader) * self.config.get('num_epochs',20) // self.config.get('grad_accum_steps',8))
        warmup_steps = int(total_steps * self.config.get('warmup_ratio',0.06))
        def lr_lambda(current_step: int):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * (current_step - warmup_steps) / (total_steps - warmup_steps))))
        self.scheduler = LambdaLR(self.optimizer, lr_lambda)
        # prepare curriculum buckets if requested
        self.curriculum_bins = self.config.get('curriculum_bins', None)
        if self.curriculum_bins is not None:
            # compute difficulty for each sample in train_ds and bucket indices
            difficulties = []
            for i in range(len(self.train_ds)):
                try:
                    item = self.train_ds[i]
                    difficulties.append(item.get('difficulty', 0.0))
                except Exception:
                    difficulties.append(0.0)
            self._build_curriculum_buckets(difficulties, self.curriculum_bins)

    def _build_curriculum_buckets(self, difficulties, bins):
        arr = np.array(difficulties)
        edges = np.linspace(arr.min(), arr.max(), bins+1)
        self.buckets = []
        for i in range(bins):
            mask = (arr >= edges[i]) & (arr <= edges[i+1])
            idxs = np.where(mask)[0].tolist()
            self.buckets.append(idxs)

    def train(self):
        self.model.train()
        num_epochs = self.config.get('num_epochs',20)
        for epoch in range(num_epochs):
            # Optionally use curriculum bucket sampling per epoch
            if hasattr(self, 'buckets') and self.buckets:
                # schedule: progress through buckets linearly across epochs
                bucket_idx = min(epoch * len(self.buckets) // max(1, num_epochs), len(self.buckets)-1)
                indices = self.buckets[bucket_idx]
                if len(indices) == 0:
                    train_loader = self.train_loader
                else:
                    sampler = SubsetRandomSampler(indices)
                    train_loader = DataLoader(self.train_ds, batch_size=self.config['batch_size'], sampler=sampler, num_workers=self.config.get('num_workers',4), pin_memory=True)
            else:
                train_loader = self.train_loader
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
            epoch_loss = 0.0
            for step, batch in enumerate(pbar):
                poses = batch['pose'].to(self.device)
                labels = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                with torch.cuda.amp.autocast(enabled=self.config.get('use_amp', True)):
                    outputs = self.model(labels=labels, attention_mask=attention_mask, pose=poses)
                    loss = outputs.get('loss', torch.tensor(0.0, device=self.device))
                    # encoder distillation loss (MSE) between teacher and student representations
                    if 'teacher_repr' in outputs and 'student_repr' in outputs and self.config.get('distill_alpha', 0.0) > 0.0:
                        teacher_repr = outputs['teacher_repr']
                        student_repr = outputs['student_repr']
                        if isinstance(teacher_repr, torch.Tensor) and isinstance(student_repr, torch.Tensor):
                            mse = nn.functional.mse_loss(student_repr, teacher_repr.to(student_repr.device))
                            loss = loss + self.config.get('distill_alpha', 1.0) * mse
                    loss = loss / max(1, self.config.get('grad_accum_steps',8))
                self.scaler.scale(loss).backward()
                if (step + 1) % self.config.get('grad_accum_steps',8) == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.get('grad_clip',1.0))
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self.scheduler.step()
                    self.global_step += 1
                epoch_loss += float(loss.item())
                pbar.set_postfix({'loss': float(loss.item())})
            val_loss = self.evaluate()
            logger.info(f"Epoch {epoch+1} train_loss={epoch_loss:.4f} val_loss={val_loss:.4f}")
            # early stopping
            if val_loss < self.best_val - 1e-4:
                self.best_val = val_loss
                self.no_improve_epochs = 0
                # save checkpoint
                ckpt_path = os.path.join(self.config.get('checkpoint_dir','checkpoints'), f'best_epoch_{epoch+1}.pt')
                torch.save({'epoch': epoch+1, 'model_state': self.model.state_dict(), 'optimizer_state': self.optimizer.state_dict()}, ckpt_path)
                logger.info(f"Saved best checkpoint to {ckpt_path}")
            else:
                self.no_improve_epochs += 1
                if self.no_improve_epochs >= self.early_stop_patience:
                    logger.info(f"Early stopping: no improvement for {self.no_improve_epochs} epochs")
                    break

    def evaluate(self):
        self.model.eval()
        total_loss = 0.0
        preds = []
        refs = []
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Eval'):
                poses = batch['pose'].to(self.device)
                labels = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                outputs = self.model(labels=labels, attention_mask=attention_mask, pose=poses)
                loss = outputs.get('loss', torch.tensor(0.0, device=self.device))
                total_loss += float(loss.item())
                encwrap = outputs.get('encoder_outputs', None)
                if encwrap is not None:
                    gen = self.model.decoder.generate(encoder_outputs=encwrap, max_length=self.config.get('max_text_len',64), num_beams=self.config.get('num_beams',3)) # Reduced num_beams
                    decoded = [self.model.decoder.config.tokenizer.decode(g, skip_special_tokens=True) for g in gen]
                    preds.extend(decoded)
                    refs.extend([b for b in batch['text']])
        avg_loss = total_loss / max(1, len(self.val_loader))
        if jiwer is not None and len(preds) > 0:
            wer = jiwer.wer(refs, preds)
            logger.info(f"Eval WER: {wer:.4f}")
        self.model.train()
        return avg_loss

def main():
    train_csv = 'E:/SignAi ODD/ai/iSign_v1.1_clean.csv'
    val_csv = 'E:/SignAi ODD/ai/iSign_val_v1.1_clean.csv'
    data_dir = 'E:/SignAi ODD/ai/pose_npys'

    if not os.path.exists(train_csv) or not os.path.exists(val_csv):
        logger.error('Missing CSV files. Put train/val CSVs next to script.')
        return
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    config = {
        'train_df': train_df,
        'val_df': val_df,
        'data_dir': data_dir,
        'tokenizer_name': 'google/flan-t5-small', # Reduced decoder model
        'num_frames': 16, # Reduced frames
        'temporal_stride': 2,
        'batch_size': 8,
        'grad_accum_steps': 8,
        'num_epochs': 20,
        'lr': 5e-5,
        'weight_decay': 1e-2,
        'warmup_ratio': 0.06,
        'use_amp': True,
        'pose_out': 128, # Reduced pose out dim
        'pose_hidden': 64, # Reduced pose hidden dim
        'd_model': 512, # Reduced d_model
        'label_smoothing': 0.1,
        'use_ctc': True,
        'ctc_weight': 0.3,
        'modality_dropout': 0.15,
        'max_text_len': 64,
        'num_workers': 4,
        'log_dir': 'logs',
        'checkpoint_dir': 'checkpoints',
        'num_beams': 3 # Reduced num_beams
    }
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    trainer = Trainer(config)
    trainer.train()

if __name__ == '__main__':
    main()


