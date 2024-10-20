import datetime
import pickle
import os
from geneformer import Classifier
from datasets import Dataset
from transformers import AutoModel

# Set up date and output directories
current_date = datetime.datetime.now()
datestamp = f"{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}{current_date.hour:02d}{current_date.minute:02d}{current_date.second:02d}"
datestamp_min = f"{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}"

output_prefix = "tf_dosage_sens_test"
output_dir = f"{datestamp}"
os.makedirs(output_dir, exist_ok=True)

# Load gene classification dictionary
with open("example_input_files_gene_classification_dosage_sensitive_tfs_dosage_sensitivity_TFs.pickle", "rb") as fp:
    gene_class_dict = pickle.load(fp)

#Initialize the classifier
cc = Classifier(
    classifier="gene",
    gene_class_dict=gene_class_dict,
    max_ncells=10_000,
    freeze_layers=4,
    num_crossval_splits=5,
    forward_batch_size=200,
    nproc=16
)

#link to dataset : https://huggingface.co/datasets/ctheodoris/Genecorpus-30M/tree/main/example_input_files/gene_classification/dosage_sensitive_tfs/gc-30M_sample50k.dataset
dataset = Dataset.from_file("data-00000-of-00001.arrow")
dataset.save_to_disk("prep_dataset")

# Prepare data for classification
cc.prepare_data(
    input_data_file="prep_dataset.dataset",
    output_directory=output_dir,
    output_prefix=output_prefix
)

# trainer = cc.train_all_data(model_directory = "Geneformer/my_model",
#                             prepared_input_data_file = f"{output_dir}/{output_prefix}_labeled.dataset",
#                             id_class_dict_file = f"{output_dir}/{output_prefix}_id_class_dict.pkl",
#                             output_directory = "Geneformer/trained_model",
#                             output_prefix = "trained_modal_files",
#                             save_eval_output = False,
#                             gene_balance = False)

# Perform validation
all_metrics = cc.validate(
    model_directory="/path/to/Geneformer",
    prepared_input_data_file=f"{output_dir}/{output_prefix}_labeled.dataset",
    id_class_dict_file=f"{output_dir}/{output_prefix}_id_class_dict.pkl",
    output_directory=output_dir,
    output_prefix=output_prefix
)

# Plot confusion matrix
cc.plot_conf_mat(
    conf_mat_dict={"Geneformer": all_metrics["conf_matrix"]},
    output_directory=output_dir,
    output_prefix=output_prefix
)

# Plot ROC curve
cc.plot_roc(
    roc_metric_dict={"Geneformer": all_metrics["all_roc_metrics"]},
    model_style_dict={"Geneformer": {"color": "red", "linestyle": "-"}},
    title="Dosage-sensitive vs -insensitive factors",
    output_directory=output_dir,
    output_prefix=output_prefix
)

# Output metrics
print(all_metrics)
