# Installation Instructions:
Below are two different methods of installation. The first is simpler, while the second allows you to use and modify the library files in-place. Note that only one method is required.

## Installation Directly from Github
```
python -m pip install git+https://github.com/uk-cliplab/representation-itl.git
```

## Editable Installation
Following this procedure, the git repository is clone and editably installed. This lets you edit or add to the library files without having to reinstall.

1) Download git repository: ```git clone https://github.com/uk-cliplab/representation-itl.git```

2) Move to folder:  ```cd representation-itl```

3) Install with pip:  ```pip install -e .```


# Import Examples:
```
import repitl.divergences as div

import repitl.kernel_utils as ku

import repitl.matrix_itl_approx as approx

import repitl.matrix_itl as itl

import repitl.difference_of_entropies as dent

import repitl.informativeness as inform
```
