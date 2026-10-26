from dm_control import suite
from custom_dmc import walker, cheetah, hopper, quadruped, pointmass
from custom_dmc.bandit import Bandit


def load(domain_name, task_name, environment_kwargs={}):

    make_fns = {
        'walker': walker,
        'cheetah': cheetah,
        'quadruped': quadruped,
        'point_mass_maze': pointmass,
        'hopper': hopper,
    }

    if (domain_name, task_name) in suite.ALL_TASKS:
        return suite.load(domain_name, task_name, environment_kwargs=environment_kwargs)
    if domain_name in make_fns:
        return make_fns[domain_name].make(task_name, environment_kwargs=environment_kwargs)
    if domain_name == 'bandit':
        return Bandit(task_name, environment_kwargs)
    raise NotImplementedError('Unknown domain: ', domain_name)
