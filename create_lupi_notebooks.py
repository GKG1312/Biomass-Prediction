import nbformat as nbf
import os

def create_training_notebook(path):
    nb = nbf.v4.new_notebook()

    # Introduction
    nb.cells.append(nbf.v4.new_markdown_cell("# Advanced Biomass Prediction: LUPI & Multi-Modal Distillation\n"
                                             "This notebook implements the training pipeline described in the 'Advanced Computer Vision Frameworks for Pasture Biomass Estimation' report.\n\n"
                                             "### Key Features:\n"
                                             "- **Backbone**: ConvNeXt V2-Base\n"
                                             "- **Tiling**: 4x 512x512 crops to preserve texture details\n"
                                             "- **Neck**: BiFPN for multi-scale feature fusion\n"
                                             "- **LUPI**: Learning Using Privileged Information (NDVI & Height)\n"
                                             "- **Knowledge Distillation**: Teacher-Student framework\n"
                                             "- **Losses**: Weighted Huber + Hierarchical Consistency + KD Loss"))

    shared_code = """
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.pointwise(self.depthwise(x))))

class BiFPNLayer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.nodes = nn.ModuleList([DepthwiseSeparableConv(channels, channels) for _ in range(3)])

    def forward(self, p3, p4, p5):
        p4_td = self.nodes[0](p4 + F.interpolate(p5, size=p4.shape[-2:]))
        p3_out = self.nodes[1](p3 + F.interpolate(p4_td, size=p3.shape[-2:]))
        p4_out = self.nodes[2](p4_td + F.interpolate(p3_out, size=p4_td.shape[-2:]))
        return p3_out, p4_out, p5

class BiomassModel(nn.Module):
    def __init__(self, model_name, feature_dim=256):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, features_only=True)
        feature_info = self.backbone.feature_info.get_dicts()
        self.align_p3 = nn.Conv2d(feature_info[-3]['num_chs'], feature_dim, 1)
        self.align_p4 = nn.Conv2d(feature_info[-2]['num_chs'], feature_dim, 1)
        self.align_p5 = nn.Conv2d(feature_info[-1]['num_chs'], feature_dim, 1)
        self.bifpn = BiFPNLayer(feature_dim)
        self.total_head = nn.Sequential(nn.Linear(feature_dim, 1), nn.Softplus())
        self.comp_head = nn.Sequential(nn.Linear(feature_dim, 4), nn.Softplus())
        self.aux_height = nn.Sequential(nn.Linear(feature_dim, 1))
        self.aux_ndvi = nn.Sequential(nn.Linear(feature_dim, 1), nn.Sigmoid())

    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        feats = self.backbone(x)
        p3 = self.align_p3(feats[-3]); p4 = self.align_p4(feats[-2]); p5 = self.align_p5(feats[-1])
        p3, p4, p5 = self.bifpn(p3, p4, p5)
        def pool(f): 
            f = f.view(b, t, *f.shape[1:]).mean(dim=1)
            return nn.AdaptiveAvgPool2d(1)(f).flatten(1)
        f3, f4, f5 = pool(p3), pool(p4), pool(p5)
        return self.total_head(f4), self.comp_head(f4), self.aux_height(f5), self.aux_ndvi(f5), f4
"""

    # Config cell
    nb.cells.append(nbf.v4.new_code_cell("""
CONFIG = {
    'seed': 42,
    'backbone': 'convnextv2_tiny.fcmae_ft_in22k_in1k',
    'img_size': 512,
    'batch_size': 4,
    'data_dir': r'..\\csiro-biomass',
    'epochs_teacher': 15,
    'epochs_distill': 25,
    'epochs_finetune': 10,
    'lr': 2e-4,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'n_splits': 3,
    'target_cols': ['Dry_Total_g', 'GDM_g', 'Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g'],
    'weights': torch.tensor([0.5, 0.2, 0.1, 0.1, 0.1]),
    'use_amp': True,
    'target_scale': 100.0
}
""".strip()))

    nb.cells.append(nbf.v4.new_code_cell(shared_code))

    nb.cells.append(nbf.v4.new_code_cell("""
class TiledBiomassDataset(Dataset):
    def __init__(self, df, img_dir, target_cols, species_map, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.target_cols = target_cols
        self.species_map = species_map
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['image_path'])
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        h, w, _ = image.shape
        tiles = [
            image[0:512, 0:512], image[0:512, w-512:w], 
            image[h-512:h, 0:512], image[h-512:h, w-512:w]
        ]
        if self.transform:
            tiles = [self.transform(image=t)['image'] for t in tiles]
        else:
            tf = A.Compose([A.Resize(512, 512), A.Normalize(), ToTensorV2()])
            tiles = [tf(image=t)['image'] for t in tiles]
        
        tiles = torch.stack(tiles)
        targets = torch.tensor(row[self.target_cols].values.astype(np.float32)) / CONFIG['target_scale']
        
        species_id = self.species_map.get(row['Species'], 0)
        meta = torch.tensor([float(row['Pre_GSHH_NDVI']), float(row['Height_Ave_cm']), float(species_id)], dtype=torch.float32)
        return tiles, targets, meta
""".strip()))

    nb.cells.append(nbf.v4.new_code_cell("""
def get_transforms(img_size, is_train=True):
    if is_train:
        return A.Compose([
            A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5),
            A.RandomShadow(p=0.3), A.ColorJitter(brightness=0.2, contrast=0.2, p=0.4),
            A.Normalize(), ToTensorV2()
        ])
    else:
        return A.Compose([A.Normalize(), ToTensorV2()])
""".strip()))

    nb.cells.append(nbf.v4.new_code_cell("""
class TeacherModel(nn.Module):
    def __init__(self, student, num_species):
        super().__init__()
        self.student = student
        self.species_embed = nn.Embedding(num_species, 32)
        self.meta_embed = nn.Sequential(
            nn.Linear(2 + 32, 64), 
            nn.ReLU(),
            nn.Linear(64, 256)
        )
    def forward(self, x, meta):
        _, _, _, _, feat = self.student(x) # feat is [B, 256]
        s_idx = meta[:, 2].long()
        s_emb = self.species_embed(s_idx)
        m_in = torch.cat([meta[:, :2], s_emb], dim=1)
        m_feat = self.meta_embed(m_in)
        f = feat + m_feat
        return self.student.total_head(f), self.student.comp_head(f), f
""".strip()))

    nb.cells.append(nbf.v4.new_code_cell("""
class HierarchicalLoss(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.weights = weights.to(CONFIG['device'])
        self.huber = nn.HuberLoss()
    def forward(self, total_p, comps_p, targets):
        # targets order: [Total, GDM, Green, Dead, Clover]
        l_total = self.huber(total_p.squeeze(), targets[:, 0])
        l_comps = self.huber(comps_p, targets[:, 1:])
        # 0: GDM, 1: Green, 2: Dead, 3: Clover in comps_p
        l_cons1 = F.mse_loss(total_p.squeeze(), comps_p[:, 1] + comps_p[:, 2]) # Total = Green + Dead
        l_cons2 = F.mse_loss(comps_p[:, 1], comps_p[:, 0] + comps_p[:, 3])    # Green = GDM + Clover
        return 0.5 * l_total + 0.5 * l_comps.mean() + 0.1 * (l_cons1 + l_cons2)
""".strip()))

    nb.cells.append(nbf.v4.new_code_cell("""
train_df = pd.read_csv(os.path.join(CONFIG['data_dir'], 'train.csv'))
df_wide = train_df.pivot_table(index=['image_path', 'Sampling_Date', 'State', 'Species', 'Pre_GSHH_NDVI', 'Height_Ave_cm'], 
                              columns='target_name', values='target').reset_index()
df_wide['group'] = df_wide['Sampling_Date'] + '_' + df_wide['State']
species_map = {s: i for i, s in enumerate(sorted(df_wide['Species'].unique()))}
num_species = len(species_map)
sgkf = StratifiedGroupKFold(n_splits=CONFIG['n_splits'])
df_wide['fold'] = -1
y_bins = pd.cut(df_wide['Dry_Total_g'], bins=10, labels=False)
for f, (t_, v_) in enumerate(sgkf.split(df_wide, y_bins, groups=df_wide['group'])):
    df_wide.loc[v_, 'fold'] = f
""".strip()))

    nb.cells.append(nbf.v4.new_code_cell("""
def train_one_fold(fold):
    train_ds = TiledBiomassDataset(df_wide[df_wide.fold != fold], CONFIG['data_dir'], CONFIG['target_cols'], species_map, get_transforms(512, True))
    val_ds = TiledBiomassDataset(df_wide[df_wide.fold == fold], CONFIG['data_dir'], CONFIG['target_cols'], species_map, get_transforms(512, False))
    loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=2)
    
    student = BiomassModel(CONFIG['backbone']).to(CONFIG['device'])
    teacher = TeacherModel(student, num_species).to(CONFIG['device'])
    
    if torch.cuda.device_count() > 1:
        teacher = nn.DataParallel(teacher)
        student_dp = nn.DataParallel(student)
    else:
        student_dp = student

    opt = torch.optim.AdamW(teacher.parameters(), lr=CONFIG['lr'])
    criterion = HierarchicalLoss(CONFIG['weights'])
    scaler = torch.amp.GradScaler('cuda', enabled=CONFIG['use_amp'])

    def validate(m, loader_v):
        m.eval(); preds, truths = [], []
        with torch.no_grad():
            for x, y, met in loader_v:
                with torch.amp.autocast('cuda', enabled=CONFIG['use_amp']):
                    out, _, _, _, _ = student_dp(x.to(CONFIG['device']))
                preds.append(out.cpu().numpy()); truths.append(y[:, 0].numpy())
        preds = np.concatenate(preds).flatten()
        truths = np.concatenate(truths).flatten()
        if len(np.unique(preds)) <= 1: return 0.0
        return np.corrcoef(preds, truths)[0,1]**2

    print(f'Training Teacher Fold {fold}...')
    for epoch in range(CONFIG['epochs_teacher']):
        teacher.train(); epoch_loss = 0
        for x, y, m in loader:
            x, y, m = x.to(CONFIG['device']), y.to(CONFIG['device']), m.to(CONFIG['device'])
            opt.zero_grad()
            with torch.amp.autocast('cuda', enabled=CONFIG['use_amp']):
                tp, cp, _ = teacher(x, m)
                loss = criterion(tp, cp, y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            epoch_loss += loss.item()
        val_r2 = validate(teacher, val_loader)
        print(f'Teacher Ep {epoch+1} | Loss: {epoch_loss/len(loader):.4f} | Val R2: {val_r2:.4f}')

    print(f'Distilling Student Fold {fold}...')
    opt_s = torch.optim.AdamW(student.parameters(), lr=CONFIG['lr'])
    for epoch in range(CONFIG['epochs_distill']):
        student.train(); epoch_loss = 0
        for x, y, m in loader:
            x, y, m = x.to(CONFIG['device']), y.to(CONFIG['device']), m.to(CONFIG['device'])
            opt_s.zero_grad()
            with torch.amp.autocast('cuda', enabled=CONFIG['use_amp']):
                with torch.no_grad(): tp_t, cp_t, f_t = teacher(x, m)
                tp_s, cp_s, h_s, n_s, f_s = student_dp(x)
                loss = criterion(tp_s, cp_s, y) + 0.5*F.mse_loss(f_s, f_t) + 0.1*F.mse_loss(h_s.squeeze(), m[:, 1]/CONFIG['target_scale'])
            scaler.scale(loss).backward(); scaler.step(opt_s); scaler.update()
            epoch_loss += loss.item()
        val_r2 = validate(student_dp, val_loader)
        print(f'Student Ep {epoch+1} | Loss: {epoch_loss/len(loader):.4f} | Val R2: {val_r2:.4f}')
    
    torch.save(student.state_dict(), f'student_fold{fold}.pth')

for f in range(CONFIG['n_splits']):
    train_one_fold(f)
""".strip()))

    with open(path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

def create_inference_notebook(path):
    nb = nbf.v4.new_notebook()
    nb.cells.append(nbf.v4.new_markdown_cell("# Advanced Biomass Prediction: TTA Inference\n"
                                             "Uses trained Student models with TTA (4 rotations)."))
    
    shared_code = """
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.pointwise(self.depthwise(x))))

class BiFPNLayer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.nodes = nn.ModuleList([DepthwiseSeparableConv(channels, channels) for _ in range(3)])

    def forward(self, p3, p4, p5):
        p4_td = self.nodes[0](p4 + F.interpolate(p5, size=p4.shape[-2:]))
        p3_out = self.nodes[1](p3 + F.interpolate(p4_td, size=p3.shape[-2:]))
        p4_out = self.nodes[2](p4_td + F.interpolate(p3_out, size=p4_td.shape[-2:]))
        return p3_out, p4_out, p5

class BiomassModel(nn.Module):
    def __init__(self, model_name, feature_dim=256):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, features_only=True)
        feature_info = self.backbone.feature_info.get_dicts()
        self.align_p3 = nn.Conv2d(feature_info[-3]['num_chs'], feature_dim, 1)
        self.align_p4 = nn.Conv2d(feature_info[-2]['num_chs'], feature_dim, 1)
        self.align_p5 = nn.Conv2d(feature_info[-1]['num_chs'], feature_dim, 1)
        self.bifpn = BiFPNLayer(feature_dim)
        self.total_head = nn.Sequential(nn.Linear(feature_dim, 1), nn.Softplus())
        self.comp_head = nn.Sequential(nn.Linear(feature_dim, 4), nn.Softplus())
        self.aux_height = nn.Sequential(nn.Linear(feature_dim, 1))
        self.aux_ndvi = nn.Sequential(nn.Linear(feature_dim, 1), nn.Sigmoid())

    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        feats = self.backbone(x)
        p3 = self.align_p3(feats[-3]); p4 = self.align_p4(feats[-2]); p5 = self.align_p5(feats[-1])
        p3, p4, p5 = self.bifpn(p3, p4, p5)
        def pool(f): 
            f = f.view(b, t, *f.shape[1:]).mean(dim=1)
            return nn.AdaptiveAvgPool2d(1)(f).flatten(1)
        f3, f4, f5 = pool(p3), pool(p4), pool(p5)
        return self.total_head(f4), self.comp_head(f4), self.aux_height(f5), self.aux_ndvi(f5), f4
"""

    nb.cells.append(nbf.v4.new_code_cell("""
CONFIG = {
    'backbone': 'convnextv2_tiny.fcmae_ft_in22k_in1k',
    'img_size': 512,
    'batch_size': 4,
    'weights': ['student_fold0.pth', 'student_fold1.pth', 'student_fold2.pth'],
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'data_dir': r'..\\csiro-biomass',
    'target_scale': 100.0
}
""".strip()))

    nb.cells.append(nbf.v4.new_code_cell(shared_code))
    
    nb.cells.append(nbf.v4.new_code_cell("""
class InferenceDataset(Dataset):
    def __init__(self, df, img_dir, rotate=0):
        self.df = df
        self.img_dir = img_dir
        self.rotate = rotate
        self.tf = A.Compose([A.Resize(512, 512), A.Normalize(), ToTensorV2()])
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(os.path.join(self.img_dir, row['image_path'])).convert('RGB')
        img = np.array(img)
        if self.rotate > 0: img = np.rot90(img, k=self.rotate)
        h, w, _ = img.shape
        tiles = [img[0:512,0:512], img[0:512,w-512:w], img[h-512:h,0:512], img[h-512:h,w-512:w]]
        return torch.stack([self.tf(image=t)['image'] for t in tiles]), row['image_path']
""".strip()))

    nb.cells.append(nbf.v4.new_code_cell("""
def run_inference():
    test_df = pd.read_csv(os.path.join(CONFIG['data_dir'], 'test.csv'))
    unique_images = test_df[['image_path']].drop_duplicates()
    target_names = ['Dry_Total_g', 'GDM_g', 'Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g']
    
    final_preds = {}
    for w_path in CONFIG['weights']:
        if not os.path.exists(w_path): continue
        print(f'Running inference for {w_path}...')
        model = BiomassModel(CONFIG['backbone']).to(CONFIG['device'])
        model.load_state_dict(torch.load(w_path, map_location=CONFIG['device']))
        model.eval()
        
        with torch.no_grad():
            for rot in [0, 1, 2, 3]:
                ds = InferenceDataset(unique_images, CONFIG['data_dir'], rotate=rot)
                loader = DataLoader(ds, batch_size=CONFIG['batch_size'])
                for x, paths in loader:
                    with torch.amp.autocast('cuda' if 'cuda' in CONFIG['device'] else 'cpu'):
                        tp, cp, _, _, _ = model(x.to(CONFIG['device']))
                    p = (torch.cat([tp, cp], dim=1) * CONFIG['target_scale']).cpu().numpy()
                    for i in range(len(paths)):
                        path = paths[i]
                        if path not in final_preds: final_preds[path] = p[i] / (len(CONFIG['weights']) * 4)
                        else: final_preds[path] += p[i] / (len(CONFIG['weights']) * 4)
    
    sub_rows = []
    for path, p in final_preds.items():
        img_id = os.path.basename(path).replace('.jpg', '')
        for i, name in enumerate(target_names):
            sub_rows.append({'sample_id': f'{img_id}__{name}', 'target': p[i]})
    pd.DataFrame(sub_rows).to_csv('submission.csv', index=False)

run_inference()
""".strip()))

    with open(path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

if __name__ == '__main__':
    os.makedirs('competition_notebooks', exist_ok=True)
    create_training_notebook('competition_notebooks/Training_LUPI_Advanced.ipynb')
    create_inference_notebook('competition_notebooks/Inference_LUPI_Advanced.ipynb')
    print("Notebooks created.")
