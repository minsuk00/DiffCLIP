import torch
import time
import datetime


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) * 100.0 / batch_size for k in topk]


@torch.no_grad()
def collect_features(model, loader, device):
    model.eval()
    # print(f"collecting features on device: {device}...")
    X, Y = [], []
    total_len = len(loader)
    start_time = time.time()
    avg_time_per_batch = 0
    for cnt, batch in enumerate(loader):
        temp_time = time.time()

        x, y = batch
        x = x.to(device)
        y = y.to(device)

        x = model(x).pooler_output
        X.append(x.detach())
        Y.append(y.detach())

        avg_time_per_batch = (avg_time_per_batch * cnt + (time.time() - temp_time)) / (cnt + 1)
        eta_seconds = (total_len - (cnt + 1)) * avg_time_per_batch
        print(
            f"collect done: {cnt+1} / {total_len}, eta: {datetime.timedelta(seconds=eta_seconds)}, time_passed: {datetime.timedelta(seconds=(time.time()-start_time))}",
            end="\r",
        )
    X = torch.cat(X).detach()
    Y = torch.cat(Y).detach()
    model.train()
    return X, Y
