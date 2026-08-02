"""Executable smoke tests for collection, dynamics, and private model rollout."""

from __future__ import annotations

import argparse, json, math
from pathlib import Path
from typing import Any
import numpy as np

from private_mb_sac.agents.networks import initialize_independent_actor_params, make_independent_actor_apply
from private_mb_sac.agents.snapshots import exchange_actor_snapshots
from private_mb_sac.core.config import clone_config, dump_config, expected_interaction_counts, load_config, resolved_run_name
from private_mb_sac.dynamics.normalization import save_actor_observation_normalizer
from private_mb_sac.envs.adapter import identity_actor_normalizer, make_private_mb_sac_adapter
from private_mb_sac.replay.buffer import create_private_replays
from private_mb_sac.rollout.real_collector import make_private_real_collector
from private_mb_sac.tracking.wandb_logger import WandbLogger
from private_mb_sac.training.calibration import calibrate_common_actor_normalizer
from private_mb_sac.training.coordinator import collect_private_round, initialize_owner_runtimes
from private_mb_sac.training.dynamics_update import initialize_owner_dynamics_runtimes, pairwise_parameter_distance, train_private_owner_dynamics
from private_mb_sac.training.model_rollout import initialize_owner_model_runtimes, generate_private_owner_model_rollout, validate_real_reward_reconstruction


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--config',type=Path,default=Path('configs/default.yaml'))
    p.add_argument('--output-dir',type=Path,default=Path('outputs/smoke'))
    p.add_argument('--smoke-test',action='store_true')
    p.add_argument('--dynamics-smoke-test',action='store_true')
    p.add_argument('--model-rollout-smoke-test',action='store_true')
    p.add_argument('--smoke-rounds',type=int,default=1)
    p.add_argument('--smoke-num-envs',type=int,default=2)
    p.add_argument('--dynamics-smoke-updates',type=int,default=10)
    p.add_argument('--dynamics-smoke-batch-size',type=int,default=32)
    p.add_argument('--model-smoke-batch-size',type=int,default=32)
    p.add_argument('--model-smoke-horizon',type=int,default=2)
    p.add_argument('--calibration-smoke-rounds',type=int,default=1)
    p.add_argument('--disable-wandb',action='store_true')
    p.add_argument('--random-actions',action='store_true')
    p.add_argument('--no-jit',action='store_true')
    return p.parse_args()


def append_jsonl(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a',encoding='utf-8') as h: h.write(json.dumps(payload,sort_keys=True,allow_nan=True)+'\n')


def finite_dynamics(m):
    keys=('dynamics/loss','dynamics/nll','dynamics/state_rmse','dynamics/position_rmse','dynamics/velocity_rmse','dynamics/epistemic_mean','dynamics/aleatoric_mean','dynamics/coverage95')
    return all(math.isfinite(float(m[k])) for k in keys)


def main():
    args=parse_args()
    selected=sum(map(int,[args.smoke_test,args.dynamics_smoke_test,args.model_rollout_smoke_test]))
    if selected!=1: raise SystemExit('Select exactly one smoke-test mode.')
    phase2=args.dynamics_smoke_test or args.model_rollout_smoke_test
    phase3=args.model_rollout_smoke_test
    config=clone_config(load_config(args.config))
    config.synchronization.rounds=args.smoke_rounds
    config.collection.num_envs_per_owner=args.smoke_num_envs
    config.collection.real_transitions_per_owner_per_round=args.smoke_num_envs*int(config.collection.rollout_length)
    if phase2:
        config.dynamics.normalization_min_samples=min(int(config.dynamics.normalization_min_samples),args.dynamics_smoke_batch_size)
        config.actor.normalization_calibration.num_envs=args.smoke_num_envs
        config.actor.normalization_calibration.rounds=args.calibration_smoke_rounds
    if phase3:
        config.model_rollout.batch_size=args.model_smoke_batch_size
        config.model_rollout.horizon_min=args.model_smoke_horizon
        config.model_rollout.horizon_max=args.model_smoke_horizon
        config.model_rollout.uncertainty_min_samples=min(int(config.model_rollout.uncertainty_min_samples),args.dynamics_smoke_batch_size)
    if args.disable_wandb:
        config.tracking.enabled=False; config.tracking.mode='disabled'
    args.output_dir.mkdir(parents=True,exist_ok=True)
    dump_config(config,args.output_dir/'resolved_config.yaml')

    adapter=make_private_mb_sac_adapter(config)
    actor,live_params=initialize_independent_actor_params(seed=int(config.experiment.seed),num_agents=3,observation_dim=18,action_dim=2,hidden_dims=config.actor.hidden_dims)
    actor_apply=make_independent_actor_apply(actor,num_agents=3)
    calibration={'sample_count':0,'environment_steps':0}
    if phase2 and bool(config.actor.normalization_calibration.enabled):
        cal_collector=make_private_real_collector(adapter,actor_apply,num_envs=int(config.actor.normalization_calibration.num_envs),rollout_length=int(config.actor.normalization_calibration.rollout_length),jit=not args.no_jit)
        normalizer,calibration=calibrate_common_actor_normalizer(adapter=adapter,collector=cal_collector,actor_params=live_params,environment_seed=int(config.experiment.seed)+5000,policy_seed=int(config.experiment.seed)+6000,num_envs=int(config.actor.normalization_calibration.num_envs),rollout_length=int(config.actor.normalization_calibration.rollout_length),rounds=int(config.actor.normalization_calibration.rounds),clip=float(config.actor.normalization_calibration.clip),minimum_std=float(config.actor.normalization_calibration.minimum_std))
        save_actor_observation_normalizer(normalizer,args.output_dir/'actor_normalization.json',sample_count=int(calibration['sample_count']))
    else: normalizer=identity_actor_normalizer(adapter)

    collector=make_private_real_collector(adapter,actor_apply,num_envs=int(config.collection.num_envs_per_owner),rollout_length=int(config.collection.rollout_length),jit=not args.no_jit)
    real_replays=create_private_replays(num_owners=3,capacity=int(config.replay.real_capacity),num_agents=3,observation_dim=18,action_dim=2,model_state_dim=18)
    owners=initialize_owner_runtimes(adapter=adapter,replays=real_replays,environment_seed=int(config.experiment.seed)+10000,num_envs=int(config.collection.num_envs_per_owner))
    dynamics=None; model_runtimes=None
    if phase2: _,dynamics=initialize_owner_dynamics_runtimes(config)
    if phase3: model_runtimes=initialize_owner_model_runtimes(config)
    phase='phase3_model_rollout_smoke' if phase3 else ('phase2_dynamics_smoke' if phase2 else 'phase1_smoke')
    logger=WandbLogger(enabled=bool(config.tracking.enabled),project=str(config.tracking.project),entity=config.tracking.entity,group=str(config.tracking.group),job_type=str(config.tracking.job_type),mode=str(config.tracking.mode),name=resolved_run_name(config)+'_'+phase,tags=list(config.tracking.tags)+[phase.replace('_','-')],config=config,output_dir=args.output_dir,save_code=bool(config.tracking.save_code))
    metrics_path=args.output_dir/'metrics.jsonl'
    if metrics_path.exists(): metrics_path.unlink()
    last=[]
    try:
        if phase2:
            logger.log_system({'calibration_env_steps':float(calibration['environment_steps']),'calibration_samples':float(calibration['sample_count']),'calibration_std_min':float(calibration['std_min']),'calibration_std_max':float(calibration['std_max'])},round_index=0,commit=True)
        for r in range(1,int(config.synchronization.rounds)+1):
            owner_metrics,system=collect_private_round(round_index=r,live_actor_params=live_params,owner_runtimes=owners,collector=collector,normalizer=normalizer,environment_seed=int(config.experiment.seed)+20000,policy_seed=int(config.experiment.seed)+30000,num_envs=int(config.collection.num_envs_per_owner),rollout_length=int(config.collection.rollout_length),use_random_actions=(args.random_actions or phase2))
            if phase2:
                for i,dr in enumerate(dynamics):
                    dm=train_private_owner_dynamics(runtime=dr,replay=owners[i].real_replay,config=config,updates=args.dynamics_smoke_updates,batch_size=args.dynamics_smoke_batch_size,seed=int(config.experiment.seed)+50000+r*100+i)
                    owner_metrics[i].update(dm)
                system['dynamics_updates_total']=float(sum(x.updates_total for x in dynamics))
            if phase3:
                bank=exchange_actor_snapshots(live_params,synchronization_round=r)
                horizon=int(config.model_rollout.horizon_max)
                for i,mr in enumerate(model_runtimes):
                    reward_metrics=validate_real_reward_reconstruction(owners[i].real_replay,config,batch_size=int(config.model_rollout.reward_validation_batch_size),seed=70000+r*100+i)
                    if reward_metrics['model/reward_reconstruction_max_abs']>float(config.model_rollout.reward_validation_tolerance):
                        raise RuntimeError(f'Owner {i} reward reconstruction mismatch: {reward_metrics}')
                    mm,_=generate_private_owner_model_rollout(runtime=mr,dynamics_runtime=dynamics[i],real_replay=owners[i].real_replay,snapshot_bank=bank,actor_apply_fn=actor_apply,actor_normalizer=normalizer,config=config,round_index=r,batch_size=args.model_smoke_batch_size,horizon=horizon,seed=80000+r*100+i,jit=not args.no_jit)
                    owner_metrics[i].update(reward_metrics); owner_metrics[i].update(mm)
                system['synthetic_transitions_round']=float(sum(m['model/generated_transitions'] for m in owner_metrics))
                system['synthetic_transitions_total']=float(sum(x.generated_total for x in model_runtimes))
                system['model_replay_size_total']=float(sum(len(x.model_replay) for x in model_runtimes))
            for i,m in enumerate(owner_metrics): logger.log_owner(i,m,round_index=r,commit=False)
            logger.log_system(system,round_index=r,commit=True)
            append_jsonl(metrics_path,{'round':r,'owners':owner_metrics,'system':system})
            last=owner_metrics
            print(f'\nRound {r}')
            for i,o in enumerate(owners):
                line=f'  owner {i}: env_steps={o.env_steps_total}, replay={len(o.real_replay)}, terminal={o.real_replay.stats.terminal_count}'
                if phase2: line+=f", dyn_updates={int(owner_metrics[i]['dynamics/updates_total'])}, state_rmse={owner_metrics[i]['dynamics/state_rmse']:.6f}"
                if phase3: line+=f", model_generated={int(owner_metrics[i]['model/generated_transitions'])}, model_replay={len(model_runtimes[i].model_replay)}"
                print(line)
            print(f"  system: env_steps={int(system['real_env_steps_total'])}")
    finally: logger.finish()

    per_owner,expected_system=expected_interaction_counts(rounds=int(config.synchronization.rounds),num_owners=3,num_envs_per_owner=int(config.collection.num_envs_per_owner),rollout_length=int(config.collection.rollout_length))
    actual=[o.env_steps_total for o in owners]
    if actual!=[per_owner]*3 or sum(actual)!=expected_system: raise RuntimeError('Interaction count mismatch.')
    terminals=[o.real_replay.stats.terminal_count for o in owners]
    expected_terminal=int(config.synchronization.rounds)*int(config.collection.num_envs_per_owner)
    if terminals!=[expected_terminal]*3: raise RuntimeError(f'Terminal mismatch: {terminals}')
    summary={'status':phase+'_passed','rounds':int(config.synchronization.rounds),'num_envs_per_owner':int(config.collection.num_envs_per_owner),'rollout_length':int(config.collection.rollout_length),'per_owner_env_steps':actual,'system_env_steps':sum(actual),'calibration_env_steps':int(calibration['environment_steps']),'replay_sizes':[len(o.real_replay) for o in owners],'terminal_counts':terminals,'replay_object_ids_unique':len({id(o.real_replay) for o in owners})==3}
    if phase2:
        if any(m.get('dynamics/skipped',1)!=0 for m in last) or not all(finite_dynamics(m) for m in last): raise RuntimeError('Dynamics validation failed.')
        summary.update({'ensemble_size_per_owner':5,'total_ensemble_members':15,'dynamics_updates_per_owner':[x.updates_total for x in dynamics],'dynamics_parameter_pairwise_l2':pairwise_parameter_distance(dynamics),'actor_normalization_file':'actor_normalization.json'})
    if phase3:
        generated=[x.generated_total for x in model_runtimes]
        maximum=int(config.synchronization.rounds)*args.model_smoke_batch_size*args.model_smoke_horizon
        if any(x<1 or x>maximum for x in generated): raise RuntimeError(f'Invalid synthetic counts: {generated}, maximum={maximum}')
        if len({id(x.model_replay) for x in model_runtimes})!=3: raise RuntimeError('Model replay sharing detected.')
        if any(float(m['model/landmark_drift_max'])!=0.0 for m in last): raise RuntimeError('Landmarks moved during synthetic rollout.')
        summary.update({'model_rollout_batch_size':args.model_smoke_batch_size,'model_rollout_horizon':args.model_smoke_horizon,'maximum_synthetic_transitions_per_owner':maximum,'synthetic_transitions_per_owner':generated,'synthetic_transitions_system':sum(generated),'model_replay_sizes':[len(x.model_replay) for x in model_runtimes],'model_replay_object_ids_unique':len({id(x.model_replay) for x in model_runtimes})==3,'owner_metrics':last})
    path=args.output_dir/f'{phase}_summary.json'; path.write_text(json.dumps(summary,indent=2,allow_nan=True),encoding='utf-8')
    print(f'\n{phase} passed.'); print(json.dumps(summary,indent=2,allow_nan=True))

if __name__=='__main__': main()
