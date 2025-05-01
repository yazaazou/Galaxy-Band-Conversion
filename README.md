# Mapping Galaxy Images Across Ultraviolet, Visible and Infrared Bands Using Generative Deep Learning

As it currently stands, the available model and training data comes from the Illustris cosmological simulations. To train a model from scratch, run the 'main.py' script in the 'python_scripts' file. Note that validation statistics will be run at the end of training on the final model.

Please specify the suffix in 'main.py', otherwise it will remain as 'replace_ME'.

Before running any of the above, you must download the desired training data. To do that, please run the shell scripts corresponding to the desired bands in the 'illustris_shell_scripts' file.

Training logs and saved models are available as well as is the 8-model ensemble used for uncertainty quantification.
