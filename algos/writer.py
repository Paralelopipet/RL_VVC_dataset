from torch.utils.tensorboard import SummaryWriter

writer = 0

def writeAgent(lagrange_multiplier, writer_counter, algo, parameter_name):
    global writer
    if writer == 0:
        writer = SummaryWriter("log/online"+algo)
    writer.add_scalar(parameter_name, lagrange_multiplier, writer_counter) 