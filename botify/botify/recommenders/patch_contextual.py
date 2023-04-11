from .random import Random
from .recommender import Recommender
import random


class PatchContextual(Recommender):
    """
    Patched Contextual recommender
    """

    def __init__(self, tracks_redis, catalog, cash):
        self.tracks_redis = tracks_redis
        self.fallback = Random(tracks_redis)
        self.catalog = catalog
        self.user_track = cash["user_track"]
        self.user_prev_tracks = cash["user_prev_tracks"]

    def _filter_previous(self, recommendations, previous, borders):
        for border in sorted(borders):
            t_recommendations = [x for x in recommendations[:border] if x not in previous]
            if t_recommendations:
                return t_recommendations
        return recommendations[:15]

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        if user not in self.user_track:
            self.user_track[user] = prev_track

        if user not in self.user_prev_tracks:
            self.user_prev_tracks[user] = {prev_track}
        else:
            self.user_prev_tracks[user].add(prev_track)

        user_track = self.tracks_redis.get(self.user_track[user])
        if user_track is None:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)

        user_track = self.catalog.from_bytes(user_track)
        recommendations = user_track.recommendations
        if not recommendations:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)

        shuffled = self._filter_previous(recommendations, self.user_prev_tracks[user], borders=(10, 25, 40, 70, 100))
        random.shuffle(shuffled)
        return shuffled[0]

