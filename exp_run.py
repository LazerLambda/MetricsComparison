from Comparison import Comparison


exp = Comparison()

perturbations = [
        'negated',
        'word_drop',
        'word_drop_every_sentence',
        'word_swap',
        'word_swap_every_sentence',
        'pos_drop_adj',
        'pos_drop_det',
        'repetitions']

metrics  = [ exp.experiment.metrics.comp_BERTScore, exp.experiment.metrics.comp_BLEURT, exp.experiment.metrics.comp_ME]
exp.set_dir().config_exp(n=2, degrees=1)
exp.create_data().evaluate(metrics)
exp.pipeline().create_plot()