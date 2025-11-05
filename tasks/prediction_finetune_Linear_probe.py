# tasks/prediction_linear_probe.py

import os
import argparse
from datetime import datetime
import time
import sys
import numpy as np

sys.path.append('.')
sys.path.append('../')
sys.path.append('../..')

import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
from torch.autograd import Variable

from src.model.PotNet.models.potnet import PotNet
from src.model.cgcnn import CrystalGraphConvNet
from src.model.matformer.models.pyg_att import Matformer
from src.model.transformer_dos import TransformerDOS
from src.model.ResNeXt_3D import resnext50

from src.data.materials_project.dataset.dataset import MatDataset
from src.utils.utils import (count_parameters, fix_seed, LRScheduler, collate, 
                             switch_mode, tensros_to_device, tensors_to_cuda,
                             create_decoder)
from src.utils.train_eval_utils import eval_encoder_decoder
from config.matformer_config import matformer_config
from config.potnet_config import potnet_config
from config.cgcnn_config import cgcnn_config
from src.data.materials_project.dataset.collate_functions import collate, collate_cgcnn

parser = argparse.ArgumentParser(description='Linear Probing - Frozen Encoder + Trainable Head')

# general
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--modalities_encoders', nargs='+', type=str, default=['crystal'], 
                    help='List of modalities for encoders')
parser.add_argument('--decoder_task', type=str, default='bandgap', 
                    help='Task for decoder (e.g., bandgap, bulk_modulus)')
parser.add_argument('--checkpoint_to_probe', type=str, required=True,
                    help='Path to pre-trained checkpoint (encoder will be frozen)')
parser.add_argument('--path_checkpoint', type=str, default='./checkpoints/')
parser.add_argument('--wandb_project_name', type=str, default='scienceclip_linear_probe')
parser.add_argument('--wandb_run_name', type=str, default='linear_probe')
parser.add_argument('--wandb_dir', type=str, default='./wandb')
parser.add_argument('--wandb_api_key', type=str, default='')

# data
parser.add_argument('--train_perc', type=int, default=70)
parser.add_argument('--val_perc', type=int, default=20)
parser.add_argument('--split_seed', type=int, default=42)
parser.add_argument('--data_path', type=str, default='./data/')

# optimization (only for the linear head)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=1e-3, 
                    help='Learning rate for linear head (higher than fine-tuning)')
parser.add_argument('--wd', type=float, default=0.0, 
                    help='Weight decay (often 0 for linear probing)')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--num_workers', type=int, default=16)
parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd', 'adamw'])

# model general
parser.add_argument('--latent_dim', type=int, default=128)

# DOS encoder params
parser.add_argument('--dim', type=int, default=128)
parser.add_argument('--depth', type=int, default=4)
parser.add_argument('--heads', type=int, default=8)
parser.add_argument('--dim_head', type=int, default=64)

# Charge density encoder params
parser.add_argument('--data_dim', type=int, default=32)
parser.add_argument('--in_channels', type=int, default=1)

# logging
parser.add_argument('--checkpoint-dir', type=Path, default='./linear_probe_saved_models/',
                    metavar='DIR', help='path to checkpoint directory')
parser.add_argument('--log-dir', type=Path, default='./linear_probe_logs/',
                    metavar='LOGDIR', help='path to tensorboard log directory')
parser.add_argument('--exp', default="linear_probe", type=str, help="Name of experiment")
parser.add_argument('--log_using', type=str, default='wandb', choices=['tb', 'wandb', 'none'])
parser.add_argument('--eval_freq', type=int, default=1)
parser.add_argument('--rank', type=int, default=0)

# architecture
parser.add_argument('--crystal_arch', type=str, default='potnet', 
                    choices=['matformer', 'cgcnn', 'potnet'])
parser.add_argument('--fc_features', type=int, default=256)
parser.add_argument('--use_final_bn', action='store_true')

# data options
parser.add_argument('--normalize_targets', action='store_true', default=True)
parser.add_argument('--non_normalize_targets', action='store_true', default=False)
parser.add_argument('--file_to_keys', type=str, default=None)
parser.add_argument('--file_to_modalities_dicts', type=str, default=None)
parser.add_argument('--use_old_split', action='store_true')

# evaluation
parser.add_argument('--eval_only', action='store_true', help='Only evaluate, no training')
parser.add_argument('--eval_ckpt', type=str, default=None, help='Checkpoint for evaluation')

# distributed training
parser.add_argument('--distribute', action='store_true')


def freeze_encoder(encoder):
    """Freeze all parameters in the encoder"""
    for param in encoder.parameters():
        param.requires_grad = False
    encoder.eval()  # Set to eval mode
    print(f"Encoder frozen. Trainable parameters: {sum(p.numel() for p in encoder.parameters() if p.requires_grad)}")


def main():
    args = parser.parse_args()
    
    local_rank = 0
    args.checkpoint_dir = args.checkpoint_dir / args.exp
    args.log_dir = args.log_dir / args.exp

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    
    if args.distribute:
        world_size = int(os.getenv('OMPI_COMM_WORLD_SIZE'))
        global_rank = int(os.getenv('OMPI_COMM_WORLD_RANK'))
        print("device count: ", torch._C._cuda_getDeviceCount())
        print("global rank: ", global_rank)
        print("world_size:", world_size)
        
        local_rank = global_rank % torch._C._cuda_getDeviceCount()

        torch.distributed.init_process_group(backend='nccl', rank=global_rank, world_size=world_size)
        print(f'GPU {global_rank} reporting in. Local rank: {local_rank}.')
        torch.distributed.barrier()

        if global_rank > 0:
            sys.stdout = open(os.devnull, 'w')
            
        args.rank = global_rank
        args.world_size = world_size
        main_worker(local_rank, args)
    else:
        print("Starting single GPU linear probing...")
        main_worker(0, args)


def main_worker(gpu, args):
    
    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True
    
    assert set(args.modalities_encoders).issubset(set(['crystal', 'dos', 'charge_density']))
    assert args.decoder_task in ['bandgap', 'eform', 'is_metal', 'efermi', 'dos', 'dielectric', 
                                 'dielectric_eig', 'bulk_modulus', 'shear_modulus', 
                                 'elastic_tensor', 'compliance_tensor']
   
    # Setup logging
    if args.rank == 0:
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if args.log_using == 'tb':
            import tensorboard_logger as tb_logger
            logger = tb_logger.Logger(logdir=args.log_dir, flush_secs=2)
        elif args.log_using == 'wandb':
            date = datetime.now().strftime("%Y-%m-%d__%H_%M_%S")
            name_date = f"{args.wandb_run_name}_{date}"

            config = vars(args)
            os.environ["WANDB_API_KEY"] = args.wandb_api_key
            os.environ["WANDB_MODE"] = 'offline'
            os.environ["WANDB_DIR"] = args.wandb_dir
            wandb.init(
                project=args.wandb_project_name, 
                name=name_date,
                config=config
            )

            os.makedirs(args.path_checkpoint, exist_ok=True)
            os.makedirs(os.path.join(args.path_checkpoint, name_date), exist_ok=True)

    # Dataset
    fix_seed(args.seed)
    print("Initializing dataset...")
    modalities_to_include = list(set(args.modalities_encoders + [args.decoder_task]))
    print("All modalities: ", modalities_to_include)
    
    if args.crystal_arch == 'matformer':
        dataset = MatDataset(
            modalities=modalities_to_include, 
            non_normalize_targets=args.non_normalize_targets, 
            data_path=args.data_path, 
            crystal_file='crystal.pt',
            file_to_keys=args.file_to_keys, 
            file_to_modalities_dicts=args.file_to_modalities_dicts, 
            mask_non_intersect=False
        )
        collate_func = collate
    elif args.crystal_arch == 'cgcnn':
        dataset = MatDataset(
            modalities=modalities_to_include, 
            non_normalize_targets=args.non_normalize_targets, 
            data_path=args.data_path, 
            crystal_file='crystal_cgcnn.pt',
            file_to_keys=args.file_to_keys, 
            file_to_modalities_dicts=args.file_to_modalities_dicts, 
            mask_non_intersect=False
        )
        collate_func = collate_cgcnn
    elif args.crystal_arch == 'potnet':
        dataset = MatDataset(
            modalities=modalities_to_include, 
            non_normalize_targets=args.non_normalize_targets, 
            data_path=args.data_path, 
            crystal_file='crystal_potnet.pt',
            file_to_keys=args.file_to_keys, 
            file_to_modalities_dicts=args.file_to_modalities_dicts, 
            mask_non_intersect=False
        )
        collate_func = collate
    
    # Get normalization stats
    if not args.non_normalize_targets:
        decoder_task_mean = dataset.mean[args.decoder_task].cuda(gpu)
        decoder_task_std = dataset.std[args.decoder_task].cuda(gpu)
    else:
        decoder_task_mean = 0
        decoder_task_std = 1
    
    # Create splits
    if args.use_old_split:
        train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [0.8, 0.0, 0.2]
        )
    else:
        snumat = 'snumat_' if 'snumat_data' in args.data_path else ''
        
        if args.file_to_keys is None:
            print(f"Saving {len(dataset)} mpid keys to {args.checkpoint_dir}")
            mpid_path = os.path.join(args.checkpoint_dir, f'{snumat}{args.decoder_task}_{len(dataset)}_keys.pt')
            mpid_mod_path = os.path.join(args.checkpoint_dir, f'{snumat}{args.decoder_task}_{len(dataset)}_modalities_dict.pt')
            torch.save(dataset.keys, mpid_path)
            torch.save(dataset.modalities_dicts, mpid_mod_path)
        
        test_perc = 100 - args.train_perc - args.val_perc
        npy_train_path = os.path.join(args.data_path, 'train_test_split', 
                                      f'{snumat}{args.decoder_task}_{args.train_perc}_{args.val_perc}_{test_perc}_train.npy')
        npy_val_path = os.path.join(args.data_path, 'train_test_split', 
                                    f'{snumat}{args.decoder_task}_{args.train_perc}_{args.val_perc}_{test_perc}_val.npy')
        npy_test_path = os.path.join(args.data_path, 'train_test_split', 
                                     f'{snumat}{args.decoder_task}_{args.train_perc}_{args.val_perc}_{test_perc}_test.npy')
        
        if os.path.exists(npy_train_path):
            train_indices = np.load(npy_train_path)
            val_indices = np.load(npy_val_path)
            test_indices = np.load(npy_test_path)
            print(f"Loaded train, val and test indices from npy files")
            assert (len(train_indices) + len(val_indices) + len(test_indices) == len(dataset))
        else:
            os.makedirs(os.path.join(args.data_path, 'train_test_split'), exist_ok=True)
            indices = np.arange(len(dataset))
            np.random.shuffle(indices)

            total_samples = len(indices)
            train_size = int(args.train_perc / 100 * total_samples)
            val_size = int(args.val_perc / 100 * total_samples)

            train_indices = indices[:train_size]
            val_indices = indices[train_size:train_size + val_size]
            test_indices = indices[train_size + val_size:]
            
            np.save(npy_train_path, train_indices)
            np.save(npy_val_path, val_indices)
            np.save(npy_test_path, test_indices)
            print("Saved train, val and test indices to npy files")
        
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)
        test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Dataloaders
    if args.distribute:
        assert args.batch_size % args.world_size == 0
        per_device_batch_size = args.batch_size // args.world_size
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=per_device_batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            sampler=train_sampler,
            collate_fn=collate_func
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=args.num_workers,
            collate_fn=collate_func
        )
    
    test_batch_size = 128 if args.distribute else args.batch_size
    tdataset = test_dataset if args.eval_only else val_dataset
    
    test_loader = torch.utils.data.DataLoader(
        tdataset,
        batch_size=test_batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers,
        collate_fn=collate_func
    )

    # Initialize encoders
    output_neurons_per_task = {
        'bandgap': 1, 'eform': 1, 'efermi': 1, 'is_metal': 1, 'dos': 601, 
        'dielectric': 9, 'dielectric_eig': 3, 'bulk_modulus': 1, 
        'shear_modulus': 1, 'elastic_tensor': 15, 'compliance_tensor': 15
    }
    
    crystal_encoder = None
    dos_encoder = None
    charge_density_encoder = None
    
    print("Initializing models...")
    if 'crystal' in args.modalities_encoders:
        if args.crystal_arch == 'matformer':
            crystal_encoder = Matformer(matformer_config).cuda(gpu)
        elif args.crystal_arch == 'potnet':
            potnet_config.embedding_dim = args.latent_dim
            potnet_config.fc_features = args.fc_features
            potnet_config.final_bn = args.use_final_bn
            crystal_encoder = PotNet(potnet_config).cuda(gpu)
        elif args.crystal_arch == 'cgcnn':
            structures = dataset[0]['crystal']
            orig_atom_fea_len = structures[0].shape[-1]
            nbr_fea_len = structures[1].shape[-1]
            crystal_encoder = CrystalGraphConvNet(
                orig_atom_fea_len, 
                nbr_fea_len,
                atom_fea_len=cgcnn_config.atom_fea_len,
                n_conv=cgcnn_config.n_conv,
                h_fea_len=cgcnn_config.h_fea_len,
                n_h=cgcnn_config.n_h,
                classification=False
            ).cuda(gpu)
    
    if 'dos' in args.modalities_encoders:
        dos_encoder = TransformerDOS(
            dim=args.dim, 
            depth=args.depth, 
            heads=args.heads, 
            dim_head=args.dim_head, 
            mlp_dim=4 * args.dim
        ).cuda(gpu)
    
    if 'charge_density' in args.modalities_encoders:
        charge_density_encoder = resnext50(embedding_dim=args.latent_dim).cuda(gpu)

    if args.distribute:
        if crystal_encoder is not None:
            crystal_encoder = nn.SyncBatchNorm.convert_sync_batchnorm(crystal_encoder)
            crystal_encoder = torch.nn.parallel.DistributedDataParallel(
                crystal_encoder, device_ids=[gpu], find_unused_parameters=True
            )
        if dos_encoder is not None:
            dos_encoder = nn.SyncBatchNorm.convert_sync_batchnorm(dos_encoder)
            dos_encoder = torch.nn.parallel.DistributedDataParallel(dos_encoder, device_ids=[gpu])
        if charge_density_encoder is not None:
            charge_density_encoder = nn.SyncBatchNorm.convert_sync_batchnorm(charge_density_encoder)
            charge_density_encoder = torch.nn.parallel.DistributedDataParallel(
                charge_density_encoder, device_ids=[gpu]
            )
    
    print("Models initialized")
    
    # Dictionary with encoders
    encoders_all = {'crystal': crystal_encoder, 'dos': dos_encoder, 'charge_density': charge_density_encoder}
    encoders = {key: encoders_all[key] for key in args.modalities_encoders}
    
    # Initialize linear heads (decoders)
    tasks_with_ReLU = []
    decoders = {
        key: create_decoder(
            args.latent_dim, 
            args.decoder_task, 
            output_neurons_per_task, 
            tasks_with_ReLU
        ).cuda(gpu) 
        for key in args.modalities_encoders
    }
    
    print("Decoders: ", decoders)
    
    if args.distribute:
        decoders = {
            key: nn.SyncBatchNorm.convert_sync_batchnorm(decoders[key]) 
            for key in args.modalities_encoders
        }
        decoders = {
            key: torch.nn.parallel.DistributedDataParallel(decoders[key], device_ids=[gpu]) 
            for key in args.modalities_encoders
        }

    # Load pre-trained encoders and FREEZE them
    if not args.eval_only:
        saved_state_dict = torch.load(args.checkpoint_to_probe, map_location=torch.device('cpu'))
        
        for modality in args.modalities_encoders:
            if not args.distribute:
                saved_state_dict[f'{modality}_state_dict'] = {
                    k.replace('module.', ''): v 
                    for k, v in saved_state_dict[f'{modality}_state_dict'].items()
                }
            
            encoders[modality].load_state_dict(saved_state_dict[f'{modality}_state_dict'])
            encoders[modality] = encoders[modality].cuda(gpu)
            
            # FREEZE THE ENCODER
            freeze_encoder(encoders[modality])
        
        print(f'Loaded and FROZE pre-trained encoders from {args.checkpoint_to_probe}')
        print("=" * 80)
        print("LINEAR PROBING MODE: Only training linear heads, encoders are frozen")
        print("=" * 80)
    
    elif args.eval_only:
        saved_state_dict = torch.load(args.eval_ckpt, map_location=torch.device('cpu'))
        for modality in args.modalities_encoders:
            if not args.distribute:
                saved_state_dict[f'{modality}_state_dict'] = {
                    k.replace('module.', ''): v 
                    for k, v in saved_state_dict[f'{modality}_state_dict'].items()
                }
                saved_state_dict['decoder_' + modality + '_state_dict'] = {
                    k.replace('module.', ''): v 
                    for k, v in saved_state_dict['decoder_' + modality + '_state_dict'].items()
                }
            encoders[modality].load_state_dict(saved_state_dict[modality + '_state_dict'])
            decoders[modality].load_state_dict(saved_state_dict['decoder_' + modality + '_state_dict'])
        print(f"Loaded checkpoint for evaluation from {args.eval_ckpt}")

    # Track gradients (only decoders)
    if args.log_using == 'wandb' and args.rank == 0:
        count = 1
        for modality in args.modalities_encoders:
            wandb.watch(decoders[modality], log='all', log_freq=100, idx=count)
            count += 1
    
    # Optimization - ONLY decoder parameters
    parameters = []
    for modality in args.modalities_encoders:
        parameters += list(decoders[modality].parameters())
    
    # Count trainable parameters
    total_params = sum(p.numel() for p in parameters if p.requires_grad)
    print(f"Total trainable parameters (linear heads only): {total_params:,}")
    
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(params=parameters, lr=args.lr, weight_decay=args.wd)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(params=parameters, lr=args.lr, weight_decay=args.wd)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params=parameters, lr=args.lr, weight_decay=args.wd, momentum=0.9)
    
    # No learning rate scheduler for linear probing (typically constant LR)
    scaler = GradScaler()
    
    # Load checkpoint if exists
    if (args.checkpoint_dir / 'checkpoint.pth').is_file() and not args.eval_only:
        ckpt = torch.load(args.checkpoint_dir / 'checkpoint.pth', map_location='cpu')
        start_epoch = ckpt['epoch']
        for modality in args.modalities_encoders:
            decoders[modality].load_state_dict(ckpt['decoder_' + modality + '_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        print(f"Loaded checkpoint. Starting from epoch {start_epoch}")
    else:
        start_epoch = 1

    # Training loop
    best_loss_dict = {modality: 999999 for modality in args.modalities_encoders}
    save_best = {}
    
    if args.eval_only:
        args.epochs = 1
    
    for e in range(start_epoch, args.epochs + 1):
        for modality in args.modalities_encoders:
            save_best[modality] = False
        
        start_time = time.time()

        # Training
        if not args.eval_only:
            # Set decoders to train mode, encoders stay in eval
            for modality in args.modalities_encoders:
                decoders[modality].train()
                encoders[modality].eval()  # Keep frozen

            if args.distribute:
                train_loader.sampler.set_epoch(e)
            
            avg_loss = 0
            
            for step, data in enumerate(train_loader, start=(e-1)*len(train_loader)):
                num_samples_batch = data[args.decoder_task].shape[0]
                optimizer.zero_grad(set_to_none=True)
                
                if args.crystal_arch == 'cgcnn':
                    crystal = data['crystal']
                    targets = data[args.decoder_task].cuda(non_blocking=True)
                    targets = targets.reshape((num_samples_batch, -1))
                    
                    crystal_var = (
                        Variable(crystal[0].cuda(non_blocking=True)),
                        Variable(crystal[1].cuda(non_blocking=True)),
                        crystal[2].cuda(non_blocking=True),
                        [crys_idx.cuda(non_blocking=True) for crys_idx in crystal[3]]
                    )

                    with autocast():
                        embeddings = {}
                        # Encoder in eval mode, no gradients
                        with torch.no_grad():
                            for modality in args.modalities_encoders:
                                z = crystal_encoder(*crystal_var)
                                embeddings[modality] = z
                        
                        all_tasks = [args.decoder_task]
                        classification_tasks = ['is_metal']
                        loss_func_tasks = {
                            task: F.mse_loss if task not in classification_tasks 
                            else F.binary_cross_entropy_with_logits 
                            for task in all_tasks
                        }

                        loss = 0
                        for modality in args.modalities_encoders:
                            predictions = decoders[modality](embeddings[modality])
                            loss += loss_func_tasks[args.decoder_task](predictions, targets)

                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                else:
                    modalities_all = args.modalities_encoders + [args.decoder_task]
                    data = tensors_to_cuda(modalities_all, data, gpu)

                    with autocast():
                        embeddings = {}
                        # Encoder in eval mode, no gradients
                        with torch.no_grad():
                            for modality in args.modalities_encoders:
                                z = encoders[modality](data[modality])
                                embeddings[modality] = z
                        
                        all_tasks = [args.decoder_task]
                        classification_tasks = ['is_metal']
                        loss_func_tasks = {
                            task: F.mse_loss if task not in classification_tasks 
                            else F.binary_cross_entropy_with_logits 
                            for task in all_tasks
                        }

                        loss = 0
                        for modality in args.modalities_encoders:
                            predictions = decoders[modality](embeddings[modality])
                            targets = (data[args.decoder_task][:, 1, :] if args.decoder_task == 'dos' 
                                     else data[args.decoder_task].reshape((num_samples_batch, -1)))
                            loss += loss_func_tasks[args.decoder_task](predictions, targets)

                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                if args.rank == 0:
                    print(f"Epoch {e:03d} | It {step + 1:03d} / {len(train_loader):03d}")
                    print(f" | Linear probe loss: {loss.item():.3f}")
                    
                    if args.log_using == 'wandb':
                        to_log_wandb = {
                            'epochs': e - 1,
                            'it': step,
                            'linear_probe/loss': loss.item(),
                            'linear_probe/learning_rate': optimizer.param_groups[0]['lr']
                        }
                        wandb.log(to_log_wandb)
                    elif args.log_using == 'tb':
                        logger.log_value('linear_probe_loss', loss.item(), step)
                        logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], step)

                    avg_loss += loss.item()
            
            if args.rank == 0 and args.log_using == 'tb':
                logger.log_value('ep_loss', avg_loss / len(train_loader), e)
                print(f'Time for epoch: {time.time() - start_time:.2f} s')

        # Evaluation
        metric_encdec = None
        if args.eval_only:
            print("EVALUATING ON TEST SET >>>>>>")
        
        if args.rank == 0 and e % args.eval_freq == 0:
            all_tasks = [args.decoder_task]
            classification_tasks = ['is_metal']
            types_of_prediction = {
                task: 'classification' if task in classification_tasks else 'regression' 
                for task in all_tasks
            }
            
            metric_encdec = eval_encoder_decoder(
                args.modalities_encoders, 
                args.decoder_task, 
                encoders, 
                decoders, 
                test_loader, 
                gpu, 
                types_of_prediction,
                decoder_task_mean,
                decoder_task_std, 
                crystal_arch=args.crystal_arch
            )

            print(f'Test metrics at epoch {e:03d} / {args.epochs:03d} | time: {time.time() - start_time:.2f} s')
            print('Linear probing evaluation:')
            
            for modality in args.modalities_encoders:
                if types_of_prediction[args.decoder_task] == 'classification':
                    line_to_print = (f'{modality} {args.decoder_task} | '
                                   f'Accuracy: {metric_encdec[modality]["accuracy"]:.4f} | '
                                   f'F1-score: {metric_encdec[modality]["f1"]:.4f}')
                else:
                    line_to_print = (f'{modality} {args.decoder_task} | '
                                   f'MSE: {metric_encdec[modality]["mse"]:.4f} | '
                                   f'MAE: {metric_encdec[modality]["mae"]:.4f}')
                print(line_to_print)
                
                if args.eval_only:
                    with open(f'./linear_probe_logs/results_{args.decoder_task}.txt', 'a+') as f:
                        f.write(args.eval_ckpt + "\n" + line_to_print + "\n\n")
            
            # Save best model
            for modality in args.modalities_encoders:
                if metric_encdec[modality]["mae"] < best_loss_dict[modality]:
                    best_loss_dict[modality] = metric_encdec[modality]["mae"]
                    save_best[modality] = True
            
            if args.eval_only:
                sys.exit()
            
            if args.log_using == 'tb':
                for t, dic in metric_encdec.items():
                    for k, v in dic.items():
                        logger.log_value(f'eval_linear_probe/{t}_{k}', float(v), e)
            elif args.log_using == 'wandb':
                metrics_wandb = {
                    'epochs': e,
                    f'eval_linear_probe/{args.decoder_task}/': metric_encdec
                }
                wandb.log(metrics_wandb)
        
        if args.rank == 0:
            # Save checkpoints (only decoders, not encoders since they're frozen)
            to_save = {}
            for modality in args.modalities_encoders:
                key = 'decoder_' + modality + '_state_dict'
                to_save[key] = decoders[modality].state_dict()
                # Optionally save frozen encoder state for completeness
                key = modality + '_state_dict'
                to_save[key] = encoders[modality].state_dict()
            
            to_save['epoch'] = e
            to_save['optimizer'] = optimizer.state_dict()
            
            if metric_encdec is not None:
                to_save['metric_encdec'] = metric_encdec

            torch.save(to_save, args.checkpoint_dir / 'checkpoint.pth')
            print(f"Saved models to {args.checkpoint_dir / 'checkpoint.pth'}")
            
            if e % 10 == 0 or e == 1:
                torch.save(to_save, args.checkpoint_dir / f'checkpoint_epoch{e}.pth')
                print(f"Saved models to {args.checkpoint_dir / f'checkpoint_epoch{e}.pth'}")
            
            for modality in args.modalities_encoders:
                if save_best[modality]:
                    torch.save(to_save, args.checkpoint_dir / 'checkpoint_best_val.pth')
                    print(f"Saved BEST model to {args.checkpoint_dir / 'checkpoint_best_val.pth'}")
    
    print(">>>> BEST LOSS DICT >>>> ")
    print(best_loss_dict)
    
    if args.rank == 0 and args.log_using == 'wandb':
        wandb.finish()


if __name__ == '__main__':
    main()
    print('Done, linear probing completed!')