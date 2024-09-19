from lang import *
from simulator import *

def main():
    prog = Prog(exprs=[
    Expr(Op.LET, ["x", Expr(Op.AND, [True, True])]),       # x = True AND True -> True
    Expr(Op.LET, ["y", Expr(Op.OR, ["x", False])]),        # y = x OR False -> True
    Expr(Op.LET, ["z", Expr(Op.NAND, [False, Expr(Op.AND, [False, False])])]),    # z = NAND(False, False) -> True
    Expr(Op.RET, ["z", "y"])                               # Return [z, y] -> [True, True]
    ])

    print(run(prog))

if __name__ == "__main__":
    main()



""" 
Plain text -> intermediate rep -> opcodes.


 """