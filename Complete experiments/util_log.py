import sys


class FileOutput:
    def __init__(self, filename):
        self.file = open(filename, 'w')
        self.original_stdout = sys.stdout

    def write(self, text):
        self.file.write(text)
        self.original_stdout.write(text)

    def flush(self):
        self.file.flush()
        self.original_stdout.flush()

    def close(self):
        self.file.close()
