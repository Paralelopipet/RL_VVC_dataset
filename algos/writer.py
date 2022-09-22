from torch.utils.tensorboard import SummaryWriter

writer = 0

def writeAgent(data, writer_counter, algo, parameter_name):
    global writer
    if writer == 0:
        writer = SummaryWriter("log/online"+algo)
    writer.add_scalar(parameter_name, data, writer_counter) 