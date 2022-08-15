from torch.utils.tensorboard import SummaryWriter

writer = 0

def writeAgent(lagrange_multiplier, writer_counter, algo):
    global writer
    if writer == 0:
        writer = SummaryWriter("log/online"+algo)
    writer.add_scalar('lagrange multiplier', lagrange_multiplier, writer_counter) 