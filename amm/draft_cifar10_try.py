import matmul_datasets as md
import vq_amm
from metrics import _compute_metrics

def _eval_amm(task, est, fixedB=True, **metrics_kwargs):
    est.reset_for_new_task()
    if fixedB:
        est.set_B(task.W_test)
    Y_hat = est.predict(task.X_test, task.W_test)
    return Y_hat


task = md.load_cifar10_tasks()

est = vq_amm.PQMatmul(ncodebooks=2,ncentroids=16)
#est.fit(task.X_train, task.W_train, Y=task.Y_train)
est.fit(task.X_train, task.W_train)
print(task.X_train.shape, task.W_train.shape)
#(50000, 512) (512, 10)
Y_hat = _eval_amm(task, est)  # (10000, 10)
metrics = _compute_metrics(task, Y_hat)

print(metrics)