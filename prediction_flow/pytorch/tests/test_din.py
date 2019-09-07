from prediction_flow.features import Number, Category, Sequence, Features
from prediction_flow.transformers.column import (
    StandardScaler, CategoryEncoder, SequenceEncoder)
from prediction_flow.pytorch import AttentionGroup, DIN


from .utils import prepare_dataloader


def test_normal():
    number_features = [
        Number('userAge', StandardScaler()),
        Number('rating', StandardScaler())]

    category_features = [
        Category('userId', CategoryEncoder(min_cnt=1)),
        Category('movieId', CategoryEncoder(min_cnt=1)),
        Category('topGenre', CategoryEncoder(min_cnt=1))]

    sequence_features = [
        Sequence('title', SequenceEncoder(sep='|', min_cnt=1)),
        Sequence('genres', SequenceEncoder(sep='|', min_cnt=1)),
        Sequence('clickedMovieIds',
                 SequenceEncoder(sep='|', min_cnt=1, max_len=5)),
        Sequence('clickedMovieTopGenres',
                 SequenceEncoder(sep='|', min_cnt=1, max_len=5))]

    attention_groups = [
        AttentionGroup(
            name='group1',
            pairs=[{'ad': 'movieId', 'pos_hist': 'clickedMovieIds'},
                   {'ad': 'topGenre', 'pos_hist': 'clickedMovieTopGenres'}],
            hidden_layers=[8, 4])]

    features = Features(
        number_features=number_features,
        category_features=category_features,
        sequence_features=sequence_features)

    dataloader, _ = prepare_dataloader(features)

    model = DIN(
        features, attention_groups=attention_groups,
        num_classes=2, embedding_size=4, hidden_layers=(16, 8),
        final_activation='sigmoid', dropout=0.3)

    model(next(iter(dataloader)))
