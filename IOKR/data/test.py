
def gen():
    lines = [
        'col1,col2\n',
        'foo,bar\n',
        'foo,baz\n',
        'bar,baz\n'
    ]
    for line in lines:
        yield line

class Reader(object):
    def __init__(self, g):
        self.g = g
    def read(self, n=0):
        try:
            return next(self.g)
        except StopIteration:
            return ''