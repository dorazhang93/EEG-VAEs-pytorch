from config import *
from models import *
from pytorch_lightning.utilities.seed import seed_everything
from dataloader import VAEDataset
import torch
from tqdm import tqdm
import numpy as np

MODEL_FILENAME = "last.ckpt"
config, out_dir, _ = load_config()
# For reproducibility
seed_everything(config['exp_params']['manual_seed'], True)

gpus = config['trainer_params']['gpus']
device = torch.device("cuda:%d" % gpus[0] if len(gpus) != 0 else "cpu")

def mse_mask(p,q,m):
    assert p.shape == q.shape
    mse = (p-q)**2
    assert mse.shape == m.shape
    return np.sum(mse*m)/np.sum(m),np.sum(m)/(m.size)


def CE_mask(p,q,m):
    assert p.shape == q.shape
    ce = -p*np.log(q)
    assert ce.shape == m.shape
    return np.sum(ce*m)/np.sum(m),np.sum(m)/(m.size)

def load_model():
    """
    load pretrained model from last.ckpt. Remove "model" in front of each parameter name.
    @return:
    @rtype:
    """
    print(config)
    checkpoint_path = str(out_dir) + f"/checkpoints/{MODEL_FILENAME}"
    meta_load = torch.load(checkpoint_path)["state_dict"]
    meta_new = {}
    for k, v in meta_load.items():
        if k.startswith("model"):
            meta_new[k[6:]] = v
    model = vae_models[config['model_params']['name']](**config['model_params']).to(device)
    model.load_state_dict(meta_new)
    return model


def inference():
    model = load_model()
    model.eval()
    # load data
    dataset = VAEDataset(**config["data_params"], pin_memory=len(gpus) != 0)
    dataset.setup("predict")
    data_loader = dataset.all_dataloader()

    reconstructs = []
    latents = []
    losses=[]
    for geno in tqdm(data_loader):
        geno = geno.to(device)
        recons, input, z, _ = model.forward(geno)
        loss = model.loss_function(recons, input,z)
        reconstructs.append(recons.detach().cpu())
        latents.append(z.detach().cpu())
        losses.append(loss["loss"].detach().cpu())
    print(f"Average loss :{np.array(losses).mean()}")
    reconstructs = np.concatenate(reconstructs, axis=0)
    latents = np.concatenate(latents, axis=0)
    np.save(out_dir / "reconstruction.npy", reconstructs)
    np.save(out_dir / "latent_features.npy", latents)


if __name__=="__main__":
    inference()




