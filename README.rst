==============
LORE Explainer
==============

LORE (LOcal Rule-based Explanations) is a model-agnostic explanator capable of producing rules to provide insight on the motivation a AI-based black box provides a specific outcome for an input instance.


The method of LORE does not make any assumption on the classifier that is used for labeling. The approach used by LORE exploits the exploration of a neighborhood of the input instance, based on a genetic algorithm to generate synthetic instances, to learn a local transparent model, which can be interpreted locally by the analyst.


.. _pyscaffold-notes:

Note
====

This project has been set up using PyScaffold 4.2.1. For details and usage
information on PyScaffold see https://pyscaffold.org/.
