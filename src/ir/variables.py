class VariableManager:
    def __init__(self):
        self.vars = {}  # Variable -> (Reservoir, index) mapping
        self.inps = {}  # Input variables
        self.fwd_vars = set()  # Forward-declared variables

    def declare_input(self, name):
        assert (
            name not in self.vars
        ), f"Cannot declare {name} as input, already variable {name} : {self.vars[name]}"
        assert (
            name not in self.inps
        ), f"Cannot create input {name}, input \
            of this name already exists."
        assert (
            name not in self.fwd_vars
        ), f"Cannot declare {name} as input, already forward-declared."
        self.inps[name] = 0

    def forward_declare(self, name):
        assert (
            name not in self.vars
        ), f"Cannot forward declare {name}, already declared as \
            variable {name} : {self.vars[name]}"
        assert (
            name not in self.fwd_vars
        ), f"Cannot forward declare {name}, already forward \
            declared."
        self.fwd_vars.add(name)

    def declare_var(self, name, val, i):
        self.vars[name] = (val, i)

    def rm_fwd_dec(self, name):
        if name in self.fwd_vars:
            self.fwd_vars.remove(name)

    def get_var(self, name):
        return self.vars.get(name)
