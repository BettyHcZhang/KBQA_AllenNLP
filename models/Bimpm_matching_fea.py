"""
BiMPM (Bilateral Multi-Perspective Matching) model implementation.
"""

from typing import Dict, Optional, List, Any

from overrides import overrides
import torch
import numpy

from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.modules import FeedForward, Seq2SeqEncoder, Seq2VecEncoder, TextFieldEmbedder
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy

from multitask.models.focalloss import FocalLoss
from multitask.modules.pos_tags_embedders import PosTagsEmbedders

from allennlp.modules.bimpm_matching import BiMpmMatching
from multitask.metrics import F1Measure


@Model.register("bimpm_matching_fea_task")
class BiMpm_Fea_Task(Model):
    """
    This ``Model`` implements BiMPM model described in `Bilateral Multi-Perspective Matching
    for Natural Language Sentences <https://arxiv.org/abs/1702.03814>`_ by Zhiguo Wang et al., 2017.
    Also please refer to the `TensorFlow implementation <https://github.com/zhiguowang/BiMPM/>`_ and
    `PyTorch implementation <https://github.com/galsang/BIMPM-pytorch>`_.

    Parameters
    ----------
    vocab : ``Vocabulary``
    text_field_embedder : ``TextFieldEmbedder``
        Used to embed the ``premise`` and ``hypothesis`` ``TextFields`` we get as input to the
        model.
    matcher_word : ``BiMpmMatching``
        BiMPM matching on the output of word embeddings of premise and hypothesis.
    encoder1 : ``Seq2SeqEncoder``
        First encoder layer for the premise and hypothesis
    matcher_forward1 : ``BiMPMMatching``
        BiMPM matching for the forward output of first encoder layer
    matcher_backward1 : ``BiMPMMatching``
        BiMPM matching for the backward output of first encoder layer
    encoder2 : ``Seq2SeqEncoder``
        Second encoder layer for the premise and hypothesis
    matcher_forward2 : ``BiMPMMatching``
        BiMPM matching for the forward output of second encoder layer
    matcher_backward2 : ``BiMPMMatching``
        BiMPM matching for the backward output of second encoder layer
    aggregator : ``Seq2VecEncoder``
        Aggregator of all BiMPM matching vectors
    classifier_feedforward : ``FeedForward``
        Fully connected layers for classification.
    dropout : ``float``, optional (default=0.1)
        Dropout percentage to use.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        If provided, will be used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 pos_tag_encoder:PosTagsEmbedders,
                 matcher_word: BiMpmMatching,
                 encoder1: Seq2SeqEncoder,
                 matcher_forward1: BiMpmMatching,
                 matcher_backward1: BiMpmMatching,
                 encoder2: Seq2SeqEncoder,
                 matcher_forward2: BiMpmMatching,
                 matcher_backward2: BiMpmMatching,
                 aggregator: Seq2VecEncoder,
                 classifier_feedforward: FeedForward,
                 fea_classifier_feedforward:FeedForward,
                 dropout: float = 0.1,
                 gamma:float = 0.5,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(BiMpm_Fea_Task, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.pos_tag_encoder = pos_tag_encoder

        self.matcher_word = matcher_word

        self.encoder1 = encoder1
        self.matcher_forward1 = matcher_forward1
        self.matcher_backward1 = matcher_backward1

        self.encoder2 = encoder2
        self.matcher_forward2 = matcher_forward2
        self.matcher_backward2 = matcher_backward2

        self.aggregator = aggregator

        matching_dim = self.matcher_word.get_output_dim() + \
                       self.matcher_forward1.get_output_dim() + self.matcher_backward1.get_output_dim() + \
                       self.matcher_forward2.get_output_dim() + self.matcher_backward2.get_output_dim()

        check_dimensions_match(matching_dim, self.aggregator.get_input_dim(),
                               "sum of dim of all matching layers", "aggregator input dim")

        self.classifier_feedforward = classifier_feedforward

        self.fea_classifier_feedforward = fea_classifier_feedforward

        self.dropout = torch.nn.Dropout(dropout)

        self.metrics = {"accuracy": CategoricalAccuracy()}

        self._f1_metric = F1Measure(vocabulary=self.vocab,label_namespace="label")

        self.gamma = gamma
        self.loss = FocalLoss(self.gamma)

        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                premise: Dict[str, torch.LongTensor],
                hypothesis: Dict[str, torch.LongTensor],
                label: torch.LongTensor = None,
                pos_tags: List[torch.LongTensor]= None,
                features: List[torch.LongTensor]= None,
                metadata: List[Dict[str, Any]] = None  # pylint:disable=unused-argument
               ) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """

        Parameters
        ----------
        premise : Dict[str, torch.LongTensor]
            The premise from a ``TextField``
        hypothesis : Dict[str, torch.LongTensor]
            The hypothesis from a ``TextField``
        label : torch.LongTensor, optional (default = None)
            The label for the pair of the premise and the hypothesis
        metadata : ``List[Dict[str, Any]]``, optional, (default = None)
            Additional information about the pair
        Returns
        -------
        An output dictionary consisting of:

        logits : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing unnormalised log
            probabilities of the entailment label.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """

        mask_premise = util.get_text_field_mask(premise)
        mask_hypothesis = util.get_text_field_mask(hypothesis)

        # embedding and encoding of the premise
        embedded_premise = self.dropout(self.text_field_embedder(premise))
        encoded_premise1 = self.dropout(self.encoder1(embedded_premise, mask_premise))
        encoded_premise2 = self.dropout(self.encoder2(encoded_premise1, mask_premise))

        # embedding and encoding of the hypothesis
        embedded_hypothesis = self.dropout(self.text_field_embedder(hypothesis))
        encoded_hypothesis1 = self.dropout(self.encoder1(embedded_hypothesis, mask_hypothesis))
        encoded_hypothesis2 = self.dropout(self.encoder2(encoded_hypothesis1, mask_hypothesis))

        matching_vector_premise: List[torch.Tensor] = []
        matching_vector_hypothesis: List[torch.Tensor] = []

        def add_matching_result(matcher, encoded_premise, encoded_hypothesis):
            # utility function to get matching result and add to the result list

            matching_result = matcher(encoded_premise, mask_premise, encoded_hypothesis, mask_hypothesis)
            matching_vector_premise.extend(matching_result[0])
            matching_vector_hypothesis.extend(matching_result[1])

        # calculate matching vectors from word embedding, first layer encoding, and second layer encoding
        add_matching_result(self.matcher_word, embedded_premise, embedded_hypothesis)
        half_hidden_size_1 = self.encoder1.get_output_dim() // 2


        add_matching_result(self.matcher_forward1,
                            encoded_premise1[:, :, :half_hidden_size_1],
                            encoded_hypothesis1[:, :, :half_hidden_size_1])

        add_matching_result(self.matcher_backward1,
                            encoded_premise1[:, :, half_hidden_size_1:],
                            encoded_hypothesis1[:, :, half_hidden_size_1:])

        half_hidden_size_2 = self.encoder2.get_output_dim() // 2
        add_matching_result(self.matcher_forward2,
                            encoded_premise2[:, :, :half_hidden_size_2],
                            encoded_hypothesis2[:, :, :half_hidden_size_2])
        add_matching_result(self.matcher_backward2,
                            encoded_premise2[:, :, half_hidden_size_2:],
                            encoded_hypothesis2[:, :, half_hidden_size_2:])

        # concat the matching vectors
        matching_vector_cat_premise = self.dropout(torch.cat(matching_vector_premise, dim=2))
        matching_vector_cat_hypothesis = self.dropout(torch.cat(matching_vector_hypothesis, dim=2))

        # aggregate the matching vectors
        aggregated_premise = self.dropout(self.aggregator(matching_vector_cat_premise, mask_premise))
        aggregated_hypothesis = self.dropout(self.aggregator(matching_vector_cat_hypothesis, mask_hypothesis))

        # the final forward layer
        logits_matching = self.classifier_feedforward(torch.cat([aggregated_premise, aggregated_hypothesis], dim=-1))

        logits = self.fea_classifier_feedforward(torch.cat([logits_matching,features],dim=-1))
        probs = torch.nn.functional.softmax(logits, dim=-1)

        output_dict = {'logits': logits, "probs": probs}
        if label is not None:
            loss = self.loss(logits, label.long().view(-1))
            for metric in self.metrics.values():
                metric(logits, label)
            output_dict["loss"] = loss

            bs = embedded_premise.size()[0]
            predictions = output_dict["probs"].cpu().data.numpy()

            predicted_tags = numpy.argmax(predictions, axis=-1)
            class_probabilities = torch.zeros([bs, self.vocab.get_vocab_size()])
            for i, tag_id in enumerate(predicted_tags):
                class_probabilities[i, tag_id] = 1

            self._f1_metric(class_probabilities, label)

        output_dict['metadata']=metadata

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Converts indices to string labels, and adds a ``"label"`` key to the result.
        """
        predictions = output_dict["probs"].cpu().data.numpy()
        argmax_indices = numpy.argmax(predictions, axis=-1)
#
        output_dict['predict_label'] = argmax_indices
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        f1_dict = self._f1_metric.get_metric(reset=reset)
        metrics_to_return = {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}
        metrics_to_return.update({
            x: y for x, y in f1_dict.items() if
            "overall" in x})
        return metrics_to_return

