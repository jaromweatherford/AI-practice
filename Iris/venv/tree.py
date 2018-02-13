class Tree:
    def __init__(self):
        self.data = None
        self.children = {}

    def __init__(self, data):
        self.data = data
        self.children = {}


    def __repr__(self):
        string = "("
        string += str(self.data)
        string += "  "
        string += str(self.children)

        string += "  "
        string += str(self.data)
        string += ")"
        return string