from .episode_runner import EpisodeRunner
from .parallel_runner import ParallelRunner
from .grf_episode_runner import EpisodeRunner as GrfEpisodeRunner


REGISTRY = {
    "grfepisode": GrfEpisodeRunner,
    "episode": EpisodeRunner,
    "parallel": ParallelRunner,
}
