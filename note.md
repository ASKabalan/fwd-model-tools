
def transfer(mesh0, mesh1, box_shape, kedges: int | float | list = None):
    pk_fn = partial(power_spectrum, box_shape=box_shape, kedges=kedges)
    ks, pk0 = pk_fn(mesh0)
    ks, pk1 = pk_fn(mesh1)
    return ks, (pk1 / pk0)**.5

ratio de puissance amplitude

def coherence(mesh0, mesh1, box_shape, kedges: int | float | list = None):
    pk_fn = partial(power_spectrum, box_shape=box_shape, kedges=kedges)
    ks, pk01 = pk_fn(mesh0, mesh1)
    ks, pk0 = pk_fn(mesh0)
    ks, pk1 = pk_fn(mesh1)
    return ks, pk01 / (pk0 * pk1)**.5

correlation => mesure pour mesurer a quel point un champ est bon

correction dans le domaine spectral alignement des phases


parameter de mon model

il faut normaliser tout en gaussian normal


flat 0.2 0.3

N(0.8 0.2)

N(0 , 1)



preconditionner prior : gaussian à gaussan (sigma * X + mu)
preconditionner prior : gaussian à flat (CDF d'une gaussienne inverse)
inverse cdf d'une gaussienne applique à uniforme(0,1) => gaussian(0,1)

gaussian(0,1) => CDF d'une gaussienne => uniforme(0,1) => uniform(0.2 , 0.3)
gaussian(0,1) => init_mesh

posterior preconditionning :

gaussian(0,1) * rescale => CDF d'une gaussienne => uniforme(0,1) => uniform(0.2 , 0.3)
gaussian(0,1) * rescale => init_mesh => gaussian(0,1)

ce rescale il faut le determiner apres un first pass avec precondionner
