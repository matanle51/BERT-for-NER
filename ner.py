import numpy as np
import torch
from nltk.corpus.reader import ConllCorpusReader
from pytorch_pretrained_bert import BertTokenizer, BertForTokenClassification
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from seqeval.metrics import accuracy_score, classification_report
from sklearn import metrics

MAX_SEN_LEN = 64
BATCH_SIZE = 32
NUM_EPOCHS = 5

ner_to_id = {}
id_to_ner = {}
ne_tags = []
ne_tags_set = set()

# Important: Cross entropy has an ignore index that help us ensure that only real labels will contribute to the
# loss later we use this ignore index as our padding label id
pad_id = CrossEntropyLoss().ignore_index


def main():
    train, val, test = load_data()
    create_ner_tags_dict()

    train_dataloader, val_dataloader, test_dataloader = preprocess_sentences(train, val, test)

    # Using the pre-trained bert for token classification
    # model = BertForNer.from_pretrained("bert-base-uncased", num_labels=len(ner_to_id))
    model = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=len(ner_to_id))

    model.to(device)

    # load optimizer parameters and create Adam optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]
    optimizer = Adam(optimizer_grouped_parameters, lr=3e-5)

    # Train & Eval
    train_loss_list, val_loss_list = [], []
    for epoch in range(NUM_EPOCHS):
        train_loss = train_model(model, train_dataloader, epoch, optimizer)
        val_loss = eval_model(model, val_dataloader, epoch)

        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        print(f'Epoch: {epoch + 1:02}, Total train Loss: {train_loss:.3f}, Val. Loss: {val_loss:3f}')

    # Test on testb data
    print(f'--- Testing on testb data ---')
    test_loss = eval_model(model, test_dataloader)
    print(f'Test loss: {test_loss:3f}')


def load_data():
    """
    Load data from files using the ConllCorpusReader and create list of dictionaries that describe the words and their
    corresponding tags for train, validation and test.
    :return: train, validation and test data.
    """
    train = convert_corpus_to_lists(ConllCorpusReader('CoNLL-2003', 'train.txt', ['words', 'pos', 'ignore', 'chunk']))
    val = convert_corpus_to_lists(ConllCorpusReader('CoNLL-2003', 'valid.txt', ['words', 'pos', 'ignore', 'chunk']))  # testa will be our val set
    test = convert_corpus_to_lists(ConllCorpusReader('CoNLL-2003', 'test.txt', ['words', 'pos', 'ignore', 'chunk']))

    return train, val, test


def convert_corpus_to_lists(corpus_data):
    """
    The the Conll corpus reader object and convert it to a more comfort list of dictionaries that describe the words
    and their corresponding tags.
    :param corpus_data: ConllCorpusReader object
    :return: data object
    """
    global ne_tags_set

    res_lsts = []
    for sent in corpus_data.iob_sents():
        if not sent:
            continue
        words, nes = [], []
        for tup in sent:
            words.append(tup[0])
            nes.append(tup[2])
        ne_tags_set.update(nes)
        res_lsts.append({'words': words, 'nes': nes})

    return res_lsts


def create_ner_tags_dict():
    """
    Create dictionaries to map id to tag and tag to id
    """
    global ne_tags_set, ner_to_id, ne_tags, id_to_ner

    ne_tags = list(ne_tags_set) + ['[CLS]', '[SEP]']
    ne_tags.sort()
    id_to_ner = {idx: tag for idx, tag in enumerate(ne_tags)}
    ner_to_id = {tag: idx for idx, tag in enumerate(ne_tags)}
    print(f'Total NER tag size: {len(ne_tags)}; Tags: {ne_tags}')


def convert_examples_to_features(examples, tokenizer, cls_token_segment_id=0, pad_token_id=0, pad_token_segment_id=0,
                                 pad_token_label_id=-100, sequence_a_segment_id=0, mask_padding_with_zero=True):
    """
    Function taken from pytorch-transformers github and modified/simplified.
    The function create a list of features where each feature is a dictionary that contains the tokenized and format
    the data in a form which is suitable for BERT:
    feature = {
        input_ids: tokenize, pad and add [CLS], [SEP].
        input_mask: let the algorithm know which locations where padded.
        segment_ids: all zeros since this is the first (and only) sentence.
        label_ids: list of ne tag ids for their equivalent input ids
    }
    :param examples: data to create features from
    :param tokenizer: bert tokenizer
    :param cls_token_segment_id: token id for CLS in segment.
    :param pad_token_id: token id for pad tokens.
    :param pad_token_segment_id: token id for segment tokens.
    :param pad_token_label_id: id of padding token. this is how the CrossEntropy knows to ignore this padding (-100 in CrossEntropy).
    :param sequence_a_segment_id: id for segment_sequence
    :param mask_padding_with_zero: do we want to pad with zeros
    :return: list of features dictionaries
    """
    cls_token = '[CLS]'
    sep_token = '[SEP]'

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens = []
        label_ids = []
        for word, label in zip(example['words'], example['nes']):
            word_tokens = tokenizer.tokenize(word)
            tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            label_ids.extend([ner_to_id[label]] + [pad_token_label_id] * (len(word_tokens) - 1))

        # Account for [CLS] and [SEP] with "- 2"
        special_tokens_count = 2
        if len(tokens) > MAX_SEN_LEN - special_tokens_count:
            tokens = tokens[:(MAX_SEN_LEN - special_tokens_count)]
            label_ids = label_ids[:(MAX_SEN_LEN - special_tokens_count)]

        tokens += [sep_token]
        label_ids += [pad_token_label_id]

        segment_ids = [sequence_a_segment_id] * len(tokens)

        tokens = [cls_token] + tokens
        label_ids = [pad_token_label_id] + label_ids
        segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = MAX_SEN_LEN - len(input_ids)

        input_ids += ([pad_token_id] * padding_length)
        input_mask += ([0 if mask_padding_with_zero else 1] * padding_length)
        segment_ids += ([pad_token_segment_id] * padding_length)
        label_ids += ([pad_token_label_id] * padding_length)

        assert len(input_ids) == MAX_SEN_LEN
        assert len(input_mask) == MAX_SEN_LEN
        assert len(segment_ids) == MAX_SEN_LEN
        assert len(label_ids) == MAX_SEN_LEN

        features.append(
            {'input_ids': input_ids, 'input_mask': input_mask, 'segment_ids': segment_ids, 'label_ids': label_ids})

    return features


def create_dataloader(data):
    """
    Create data-loader for data using the corresponding features extracted from the data (input_ids, input_mask,
    segment_ids, label_ids).
    :param data: list of data that contains the features
    :return: data-loader
    """
    input_ids = torch.LongTensor([sent['input_ids'] for sent in data])
    input_mask = torch.LongTensor([sent['input_mask'] for sent in data])
    segment_ids = torch.LongTensor([sent['segment_ids'] for sent in data])
    label_ids = torch.LongTensor([sent['label_ids'] for sent in data])

    dataset = TensorDataset(input_ids, input_mask, segment_ids, label_ids)

    train_sampler = RandomSampler(dataset)
    dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=BATCH_SIZE)

    return dataloader


def preprocess_sentences(train, val, test):
    """
    Tokenize, extract features and create data-loaders for train, validation and test datasets.
    :param train: train data
    :param val: validation data
    :param test: test data
    :return: train, validation and test data-loaders
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    train_data = convert_examples_to_features(train, tokenizer, pad_token_label_id=pad_id)
    val_data = convert_examples_to_features(val, tokenizer, pad_token_label_id=pad_id)
    test_data = convert_examples_to_features(test, tokenizer, pad_token_label_id=pad_id)

    train_dataloader = create_dataloader(train_data)
    val_dataloader = create_dataloader(val_data)
    test_dataloader = create_dataloader(test_data)

    return train_dataloader, val_dataloader, test_dataloader


def train_model(model, train_dataloader, epoch, optimizer):
    """
    Train loop - Gets a pretrained BERT model and a data-loader and fine-tune it.
    :param model: pre-trained model
    :param train_dataloader: train data-loader
    :param epoch: epoch number
    :param optimizer: optimizer
    """
    model.train()  # Set model to train mode
    total_epoch_loss = 0
    steps = 0

    for idx, (input_ids, input_mask, segment_ids, label_ids) in enumerate(train_dataloader):
        input_ids, input_mask, segment_ids, label_ids = input_ids.to(device), input_mask.to(device), segment_ids.to(device), label_ids.to(device)
        loss = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=label_ids)
        loss.backward()
        total_epoch_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=1)
        optimizer.step()
        model.zero_grad()

        steps += 1

        if steps % 100 == 0:
            print(f'Epoch: {epoch + 1}, Iteration: {(idx + 1)}/{len(train_dataloader)}, Training Loss: {total_epoch_loss/steps:.4f}')

    return total_epoch_loss/len(train_dataloader)


def eval_model(model, dataloader, epoch=-1):
    """
    Evaluation loop - Gets model and a data-loader and extract statistics for the specific evaluation epoch
    :param model: model to evaluate on
    :param dataloader: data loader
    :param epoch: epoch number
    """
    predicted_ne, true_ne = [], []
    total_epoch_loss = 0
    total_epoch_acc = 0

    model.eval()
    with torch.no_grad():
        for idx, (input_ids, input_mask, segment_ids, label_ids) in enumerate(dataloader):
            input_ids, input_mask, segment_ids, label_ids = input_ids.to(device), input_mask.to(device), segment_ids.to(device), label_ids.to(device)
            loss = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=label_ids)
            logits = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=None)

            total_epoch_loss += loss

            iter_pred_ne = np.argmax(logits.data.cpu().numpy(), axis=2)
            iter_true_ne = label_ids.data.cpu().numpy()

            true_ne.append(iter_true_ne)
            predicted_ne.extend([list(max_res) for max_res in iter_pred_ne])

            acc = np.sum(iter_pred_ne.flatten() == iter_true_ne.flatten())/len(iter_pred_ne.flatten())
            total_epoch_acc += acc

    # Create list of lists with true and predicted tags for easy comparision
    flatten_true_ne = [[ne_id for ne_id in col] for entry in true_ne for col in entry]
    pred_tags = [[id_to_ner[ne_id] for j, ne_id in enumerate(sent) if flatten_true_ne[i][j] != pad_id] for i, sent in enumerate(predicted_ne)]
    true_tags = [[id_to_ner[ne_id] for ne_id in sent if ne_id != pad_id] for sent in flatten_true_ne]
    assert all([len(x) == len(y) for x, y in zip(pred_tags,true_tags)])  # Confirm all lists are of same length

    stats_per_tag(pred_tags, true_tags, epoch)
    seqeval_report(pred_tags, true_tags, epoch)

    return total_epoch_loss/len(dataloader)


def seqeval_report(pred_tags, true_tags, epoch):
    """
    when using the predicted named-entities for downstream tasks,
    it is more useful to evaluate with metrics at a full named-entity level.
    http://www.davidsbatista.net/blog/2018/05/09/Named_Entity_Evaluation/
    :param pred_tags: predicted tags
    :param true_tags: true tags
    :param epoch: epoch number
    """
    report = classification_report(true_tags, pred_tags, digits=4)

    prefix = f'Val. Epoch: {epoch + 1},' if epoch + 1 > 0 else 'Test Epoch: '
    print(f'\n{prefix} seqeval results - Exact match (for all words of a chunk) as used in the official comparision:')
    print(f'{report}\n')


def stats_per_tag(pred_tags, true_tags, epoch):
    """
    For basic evaluation, we want to check all tags' results (Note - this is not the offical way a system is checked
    on the CoNLL-2003).
    :param pred_tags: predicted tags
    :param true_tags: true tags
    :param epoch: epoch number
    """
    pred_tags_flatten = [item for sublist in pred_tags for item in sublist]
    true_tags_flatten = [item for sublist in true_tags for item in sublist]

    for tag in ne_tags:
        curr_true_tag = [1 if curr_tag == tag else 0 for curr_tag in true_tags_flatten]
        curr_pred_tag = [1 if curr_tag == tag else 0 for curr_tag in pred_tags_flatten]

        acc = metrics.accuracy_score(curr_true_tag, curr_pred_tag)
        f1 = metrics.f1_score(curr_true_tag, curr_pred_tag)
        recall = metrics.recall_score(curr_true_tag, curr_pred_tag)
        precision = metrics.precision_score(curr_true_tag, curr_pred_tag)

        try:
            tn, fp, fn, tp = metrics.confusion_matrix(curr_true_tag, curr_pred_tag).ravel()
            conf_str = f'conf matrix: tn={tn}, fp={fp}, fn={fn}, tp={tp}'
        except ValueError as e:
            conf_str = 'conf matrix: No such label in dataset'

        prefix = f'Val. Epoch: {epoch + 1},' if epoch + 1 > 0 else 'Test Epoch: '
        print(f'{prefix} Tag: {tag}, accuracy: {round(acc, 4)}, '
              f'f1_score: {round(f1, 4)}, recall: {round(recall, 4)}, precision: {round(precision, 4)}, {conf_str}')


if __name__ == '__main__':
    device = torch.device('cuda')
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # For some reason it fails on my pc
    main()
