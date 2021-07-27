# What does LIME really see in images?

Python code for the paper [What does LIME really see in images?](https://arxiv.org/abs/2102.06307) No installation required, the main requirements are LIME (tested with version 0.2.0.0) and Tensorflow (>=2.1.0).

## General organization

The script ``shape_detector.py`` produces the plots for Figure 3, ``linear_model.py`` produces Figure 4. Other scripts have to run in a certain order (explained below). Consider running ``qualitative_results.py'' to produce figures for specific images. Depending on your hardware, some experiments may take time (especially computing empirical LIME explanations). You can modify the value of ``n_images`` accordingly. All auxilliary functions are stored in the ``utils'' folder, while the models used for the CIFAR-10 experiments are saved as h5 files in the ``models'' folder.

## Experiments with CIFAR-10 images

 - run ``train_model.py`` to train the models, or directly use the provided h5 files
 - run ``compute_empirical.py`` to get empirical LIME explanations
 - run ``compute_approx.py`` to get approximated explanations
 - run ``compare_exp.py`` to compare explanations

## Experiments with ILSVRC2017 images

 - download the archive <http://image-net.org/image/ILSVRC2017/ILSVRC2017_DET_test_new.tar.gz>
 - extract the archive and modify the ``data_path`` variable in the scripts accordingly
 - run ``compute_empirical.py`` to get empirical LIME explanations
 - run ``compute_approx.py`` to get approximated explanations
 - run ``compare_exp.py`` to compare explanations 

## Citing this work

If you use this software please cite 
```
@InProceedings{garreau2021what,
  title = 	 {What does LIME really see in images?},
  author =       {Garreau, Damien and Mardaoui, Dina},
  booktitle = 	 {Proceedings of the 38th International Conference on Machine Learning},
  pages = 	 {3620--3629},
  year = 	 {2021},
  editor = 	 {Meila, Marina and Zhang, Tong},
  volume = 	 {139},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {18--24 Jul},
  publisher =    {PMLR}
}
```