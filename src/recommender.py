from surprise import Dataset, Reader, KNNBasic
import pandas as pd

class Recommender:
    def __init__(self):
        self.model = None

    def fit(self, ratings_df):
        reader = Reader(rating_scale=(ratings_df['rating'].min(), ratings_df['rating'].max()))
        data = Dataset.load_from_df(ratings_df[['user', 'item', 'rating']], reader)
        trainset = data.build_full_trainset()
        self.model = KNNBasic()
        self.model.fit(trainset)

    def get_top_n(self, user_id, n=5):
        # Exemplo: retorna os n itens melhor avaliados para o usuário
        trainset = self.model.trainset
        items = trainset.all_items()
        item_inner_ids = [trainset.to_raw_iid(i) for i in items]
        predictions = [self.model.predict(user_id, iid) for iid in item_inner_ids]
        predictions.sort(key=lambda x: x.est, reverse=True)
        return [(pred.iid, pred.est) for pred in predictions[:n]]