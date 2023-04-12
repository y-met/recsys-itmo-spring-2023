from .random import Random
from .recommender import Recommender
import random
import numpy as np


class UserTrackEMBD(Recommender):
    """
    Recommend tracks closest to the previous one.
    Fall back to the random recommender if no
    recommendations found for the track.
    """

    def __init__(self, tracks_redis, catalog, embeddings):
        self.tracks_redis = tracks_redis
        self.fallback = Random(tracks_redis)
        self.catalog = catalog
        self.embeddings = embeddings

    def _filter_previous(self, recommendations, previous, borders):
        for border in sorted(borders):
            t_recommendations = [x for x in recommendations[:border] if x not in previous]
            if t_recommendations:
                return t_recommendations
        return recommendations[:15]

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        if user > len(self.embeddings["user"]):
            return self.fallback.recommend_next(user, prev_track, prev_track_time)

        previous_track = self.tracks_redis.get(prev_track)
        if previous_track is None:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)

        embedding = np.hstack((
            self.embeddings["user"][user],
            self.embeddings["context"][prev_track]))

        k = 40
        recommendations = np.argpartition(-np.dot(self.embeddings["track"], embedding), k)[:k].tolist()
        recommendations = [x for x in recommendations if x != prev_track]
        random.shuffle(recommendations)
        return recommendations[0]

