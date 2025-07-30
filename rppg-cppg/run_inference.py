import subprocess
#自建调用joint_inference
for epoch in range(30):
    e = epoch * 100
    # 移除 d=0
    cmd = f"python joint_inference.py with train_exp_num=1 e={e}"
    subprocess.run(cmd, shell=True)