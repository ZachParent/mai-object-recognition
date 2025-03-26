# 2025-03-25

## Questions for Prof

- How should we preprocess the annotations?
    - Should we use a subset of the annotations?
- For the experiments, we intend to run them serially, for each model. How does this look?
    - Is grid search necessary for any of these?
- For the final step when addressing data imbalance, does "finetune" the model mean we use the best model from the previous step? Or simply begin with pretrained model, like the other experiments?


## Answers

- How should we preprocess the annotations?
    - Should we use a subset of the annotations?
        > use first 27 categories
- For the experiments, we intend to run them serially, for each model. How does this look?
    > looks good

    > the resolution experiment is last because it is so expensive
- For the final step when addressing data imbalance, does "finetune" the model mean we use the best model from the previous step? Or simply begin with pretrained model, like the other experiments?
    > fine tuning means training the weights, we already know the hyperparams. sometimes an overrepresented class has poor metrics in comparison to the others. discard the overperforming class, and continue training

    > we are creating a new model that is only able to classify the underperforming classes.

    > how many should we discard? we will find an obvious separation.

- how many should we discard?
    > we will find an obvious separation.

- how many epochs?
    > no early stopping

    > use smaller epoch count until last exp



let's not make finetune programmatic.

let's save the best model weights, then load them for finetuning.
