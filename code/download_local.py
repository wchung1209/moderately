from transformers import DistilBertModel, DistilBertTokenizerFast

DistilBertModel.from_pretrained("distilbert-base-uncased")
DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
