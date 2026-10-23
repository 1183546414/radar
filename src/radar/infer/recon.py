# infer_recon.py  (Python 3.7+)
import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import importlib
import importlib.util
import sys

def load_statevae_from_code(code_path_or_module: str):
    """
    支持两种方式：
    1) 传入模块名，例如 'train_state_vae'（确保该文件在 PYTHONPATH 下）
    2) 传入文件路径，例如 './train_state_vae.py'
    返回模块对象（需包含 StateVAE 类）
    """
    if code_path_or_module.endswith(".py") or os.path.sep in code_path_or_module:
        path = os.path.abspath(code_path_or_module)
        name = os.path.splitext(os.path.basename(path))[0]
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        return mod
    else:
        return importlib.import_module(code_path_or_module)

def recon_main():
    from radar.paths import DATA_V2_PROCESSED, OUT_STATE_VAE, PROJECT_ROOT
    default_code = PROJECT_ROOT / "src" / "radar" / "train" / "state_vae.py"
    parser = argparse.ArgumentParser(description="Generate S_rec_std.npy using trained VAE.")
    parser.add_argument("--data", default=str(DATA_V2_PROCESSED / "S_std.npy"), help="Path to S_std.npy (shape [N, D])")
    parser.add_argument("--model", default=str(OUT_STATE_VAE / "state_vae.pt"), help="Path to state_vae.pt (state_dict)")
    parser.add_argument("--code", default=str(default_code), help="Path or module name to state_vae.py containing StateVAE")
    parser.add_argument("--latent_dim", type=int, default=16, help="latent_dim used in training (default 16)")
    parser.add_argument("--hidden",     type=int, default=128, help="hidden size used in training (default 128)")
    parser.add_argument("--batch_size", type=int, default=4096, help="Inference batch size")
    parser.add_argument("--device",     type=str, default="auto", choices=["auto","cpu","cuda"], help="Device")
    parser.add_argument("--out_recon",  type=str, default=str(DATA_V2_PROCESSED / "S_rec_std.npy"), help="Output path for reconstructed S")
    parser.add_argument("--out_mu",     type=str, default=None, help="Optional: save mu to .npy")
    parser.add_argument("--out_logvar", type=str, default=None, help="Optional: save logvar to .npy")
    parser.add_argument("--out_z",      type=str, default=None, help="Optional: save sampled z to .npy")
    parser.add_argument("--metrics",    type=str, default="recon_metrics.txt", help="Where to write MSE/RMSE")
    args = parser.parse_args()

    # 1) 载入数据
    S = np.load(args.data)
    if S.ndim == 1:
        S = S[:, None]
    if S.ndim != 2:
        raise ValueError(f"S should be 2D array, got shape {S.shape}")
    N, D = S.shape
    print(f"[INFO] Loaded S_std: shape={S.shape}")

    # 2) 选择设备
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"[INFO] Using device: {device}")

    # 3) 动态导入 StateVAE
    mod = load_statevae_from_code(args.code)
    assert hasattr(mod, "StateVAE"), "StateVAE class not found in the provided code module."
    StateVAE = getattr(mod, "StateVAE")

    # 4) 构建模型并加载权重
    model = StateVAE(s_dim=D, latent_dim=args.latent_dim, hidden=args.hidden)
    sd = torch.load(args.model, map_location="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print("[WARN] Missing keys when loading state_dict:", missing)
    if unexpected:
        print("[WARN] Unexpected keys in state_dict:", unexpected)
    model.to(device)
    model.eval()
    print("[INFO] Model loaded.")

    # 5) 推理
    ds = TensorDataset(torch.from_numpy(S).float())
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    S_rec_all = np.empty_like(S, dtype=np.float32)
    mu_all, logvar_all, z_all = [], [], []

    idx_base = 0
    with torch.no_grad():
        for (xb,) in dl:
            xb = xb.to(device)
            # 假设 forward 返回 (s_rec, mu, logvar, z)
            out = model(xb)
            if isinstance(out, (list, tuple)) and len(out) >= 4:
                s_rec, mu, logvar, z = out[:4]
            else:
                raise RuntimeError("Expected model(x) to return (s_rec, mu, logvar, z).")

            bs = xb.size(0)
            S_rec_all[idx_base:idx_base+bs] = s_rec.detach().cpu().numpy()
            if args.out_mu is not None:
                mu_all.append(mu.detach().cpu().numpy())
            if args.out_logvar is not None:
                logvar_all.append(logvar.detach().cpu().numpy())
            if args.out_z is not None:
                z_all.append(z.detach().cpu().numpy())
            idx_base += bs

    # 6) 保存输出
    np.save(args.out_recon, S_rec_all)
    print(f"[OK] Saved recon -> {args.out_recon}")

    if args.out_mu is not None:
        np.save(args.out_mu, np.concatenate(mu_all, axis=0))
        print(f"[OK] Saved mu -> {args.out_mu}")
    if args.out_logvar is not None:
        np.save(args.out_logvar, np.concatenate(logvar_all, axis=0))
        print(f"[OK] Saved logvar -> {args.out_logvar}")
    if args.out_z is not None:
        np.save(args.out_z, np.concatenate(z_all, axis=0))
        print(f"[OK] Saved z -> {args.out_z}")

    # 7) 计算并记录指标（标准化尺度上的误差）
    mse = float(np.mean((S - S_rec_all) ** 2))
    rmse = float(np.sqrt(mse))
    with open(args.metrics, "w", encoding="utf-8") as f:
        f.write(f"MSE={mse:.8f}\nRMSE={rmse:.8f}\n")
    print(f"[METRIC] MSE={mse:.6f}, RMSE={rmse:.6f}  (on standardized scale)")
    print("[DONE]")

if __name__ == "__main__":
    recon_main()
