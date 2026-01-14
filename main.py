from rsp_cegis.cegis  import CEGIS
from rsp_cegis.form   import CompleteOrderForm
from rsp_cegis.pr_ast import set_symbol_universe
import math

if __name__ == "__main__":
    form = CompleteOrderForm()
    set_symbol_universe(form.symbol_set())

    cegis = CEGIS(
        form,
        max_rounds = 100,
        starting = 10,
        generations = 30,
        elite = 5,
        target_solutions=math.inf,
    )
    rule = cegis.synthesise()
    print("\nVerified Rules:")
    for r in rule:
        print(f"- {r}")