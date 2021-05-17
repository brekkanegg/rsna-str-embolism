import sys
import time
import numpy as np

# from tensorboardX import SummaryWriter


# http://stackoverflow.com/questions/34950201/pycharm-print-end-r-statement-not-working
class Logger(object):
    def __init__(self, swidth=16):
        self.terminal = sys.stdout  # stdout
        self.file = None
        self.swidth = swidth

    def open(self, file, mode=None):
        if mode is None:
            mode = "w"
        self.file = open(file, mode)

    def write(self, message, is_terminal=1, is_file=1):
        if "\r" in message:
            is_file = 0

        if is_terminal == 1:
            self.terminal.write(message)
            self.terminal.flush()
            # time.sleep(1)

        if is_file == 1:
            self.file.write(message)
            # self.file.flush()

    def flush(self):

        pass

    def set_header_columns(self, header_columns):
        self.header_columns = header_columns

    # TODO:

    def log_header(self):
        # swidth = 16
        log_messages = (" {: ^{w}} |" * len(self.header_columns)).format(
            *self.header_columns, w=self.swidth
        )
        self.write(log_messages)
        self.write("\n")
        self.write("-" * len(log_messages) + "\n")

    def log_result(self, result):
        print("\r", end="", flush=True)
        log_messages = ""

        # FIXME: String, Integer, Time, Float
        # swidth = 16
        for r in result:
            if (type(r) == float) or (type(r) == np.float64):
                log_messages += " {: ^{w}.4f} |".format(r, w=self.swidth)
            else:
                log_messages += " {: ^{w}} |".format(r, w=self.swidth)

        self.write(log_messages)
