import wandb

ENTITY = "yachen-tian-rwth-aachen-university"
PROJECT = "AORPO-dynamics model"
RUN_ID = "oia8v9r5"   # 换成你自己的 run id

api = wandb.Api()
run = api.run(f"{ENTITY}/{PROJECT}/{RUN_ID}")

print("RUN NAME:", run.name)
print("\nSUMMARY KEYS:")
print(list(run.summary.keys()))

hist = run.history(pandas=True)
print("\nHISTORY COLUMNS:")
print(hist.columns.tolist())

print("\nHEAD:")
print(hist.head())