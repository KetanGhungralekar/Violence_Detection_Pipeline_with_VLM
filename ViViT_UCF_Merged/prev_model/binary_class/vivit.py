import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from module import Attention, PreNorm, FeedForward
import numpy as np
from torchvision import transforms
import cv2

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x, return_attention=False):
        attn_weights = []
        for attn, ff in self.layers:
            attn_output, weights = attn(x, return_attention=True)
            x = attn_output + x
            x = ff(x) + x
            if return_attention:
                attn_weights.append(weights)
        if return_attention:
            return self.norm(x), attn_weights
        return self.norm(x)
    


  
class ViViT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, num_frames, dim = 192, depth = 4, heads = 3, pool = 'cls', in_channels = 3, dim_head = 64, dropout = 0.1,
                 emb_dropout = 0.1, scale_dim = 4, ):
        super().__init__()
        
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'


        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size ** 2
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim, dim),
            nn.Dropout(dropout),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, num_patches + 1, dim))
        self.space_token = nn.Parameter(torch.randn(1, 1, dim))
        self.space_transformer = Transformer(dim, depth, heads, dim_head, dim*scale_dim, dropout)

        self.temporal_token = nn.Parameter(torch.randn(1, 1, dim))
        self.temporal_transformer = Transformer(dim, depth, heads, dim_head, dim*scale_dim, dropout)

        self.dropout = nn.Dropout(emb_dropout)
        self.pool = pool

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Dropout(dropout),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x, return_attention=False):
            x = self.to_patch_embedding(x)
            b, t, n, _ = x.shape

            cls_space_tokens = repeat(self.space_token, '() n d -> b t n d', b=b, t=t)
            x = torch.cat((cls_space_tokens, x), dim=2)
            x += self.pos_embedding[:, :, :(n + 1)]
            x = self.dropout(x)

            x = rearrange(x, 'b t n d -> (b t) n d')
            if return_attention:
                x, space_attn_weights = self.space_transformer(x, return_attention=True)
            else:
                x = self.space_transformer(x)
            x = rearrange(x[:, 0], '(b t) ... -> b t ...', b=b)

            cls_temporal_tokens = repeat(self.temporal_token, '() n d -> b n d', b=b)
            x = torch.cat((cls_temporal_tokens, x), dim=1)

            if return_attention:
                x, temporal_attn_weights = self.temporal_transformer(x, return_attention=True)
            else:
                x = self.temporal_transformer(x)

            x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
            output = self.mlp_head(x)

            if return_attention:
                return output, (space_attn_weights, temporal_attn_weights)
            return output
    
    
    
# Function to load and preprocess a video
def load_video(video_path, num_frames=16, image_size=224):
    # Try to open the video file.
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Unable to open video {video_path}. Returning zeros.")
            return torch.zeros(1, num_frames, 3, image_size, image_size)
    except Exception as e:
        print(f"Exception while opening video {video_path}: {e}. Returning zeros.")
        return torch.zeros(1, num_frames, 3, image_size, image_size)
    
    frames = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if frame_count <= 0:
        print(f"Warning: Video {video_path} has no frames. Returning zeros.")
        cap.release()
        return torch.zeros(1, num_frames, 3, image_size, image_size)
    
    # Generate sample indices (using max to avoid division by zero)
    sample_indices = np.linspace(0, max(frame_count - 1, 1), num_frames, dtype=int)
    
    for idx in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Failed to read frame {idx} from video {video_path}.")
            continue
        if idx in sample_indices:
            try:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (image_size, image_size))
            except Exception as e:
                print(f"Error processing frame {idx} from video {video_path}: {e}")
                continue
            frames.append(frame)
    
    cap.release()
    
    # Define transformation pipeline.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.3763090481649224, 0.36528133079917907, 0.3509803885002057], std=[0.2708064877155682, 0.266584700847205, 0.26854724531039814])
    ])
    
    # Check if any frames were successfully extracted.
    if len(frames) == 0:
        print(f"Warning: No frames extracted from video {video_path}. Returning zeros.")
        return torch.zeros(1, num_frames, 3, image_size, image_size)
    
    try:
        frames = torch.stack([transform(frame) for frame in frames])  # Shape: [T, C, H, W]
    except Exception as e:
        print(f"Error stacking frames from video {video_path}: {e}. Returning zeros.")
        return torch.zeros(1, num_frames, 3, image_size, image_size)
    
    # If fewer frames than expected, pad with zeros.
    if frames.shape[0] < num_frames:
        missing = num_frames - frames.shape[0]
        pad_tensor = torch.zeros(missing, 3, image_size, image_size)
        frames = torch.cat([frames, pad_tensor], dim=0)
    
    # Add batch dimension: final shape [1, T, C, H, W]
    frames = frames.unsqueeze(0)
    return frames

def load_video_new(video_path, num_frames=16, image_size=224):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Unable to open video {video_path}. Returning zeros.")
            return torch.zeros(num_frames, 3, image_size, image_size)
    except Exception as e:
        print(f"Exception while opening video {video_path}: {e}. Returning zeros.")
        return torch.zeros(num_frames, 3, image_size, image_size)
    
    frames = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_count <= 0:
        print(f"Warning: Video {video_path} has no frames. Returning zeros.")
        cap.release()
        return torch.zeros(num_frames, 3, image_size, image_size)

    # Read all frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        try:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (image_size, image_size))
            frames.append(frame)
        except Exception as e:
            print(f"Error processing frame from video {video_path}: {e}")
            continue
    
    cap.release()

    # Define transformation pipeline (only ToTensor, let the augmentation pipeline handle normalization)
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Convert to tensor
    if len(frames) == 0:
        print(f"Warning: No frames extracted from video {video_path}. Returning zeros.")
        return torch.zeros(num_frames, 3, image_size, image_size)
    
    # Convert all frames to tensor
    frames = torch.stack([transform(frame) for frame in frames])  # Shape: [T, C, H, W]

    return frames  # Return shape: [T, C, H, W] where T is the actual number of frames


if __name__ == "__main__":
    
    img = torch.ones([1, 16, 3, 224, 224]).cuda()
    
    model = ViViT(224, 16, 100, 16).cuda()
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3fM' % parameters)
    
    out = model(img)
    
    print("Shape of out :", out.shape)      # [B, num_classes]
    video_path = "wrestling.mp4"  # Change this to your video file path
    
    model = ViViT(224, 16, 100, 16).cuda()
    model.eval()
    
    video_tensor = load_video(video_path).cuda()
    with torch.no_grad():
        output = model(video_tensor)
    
    predicted_class = torch.argmax(output, dim=1).item()
    print(f"Predicted class: {predicted_class}")
