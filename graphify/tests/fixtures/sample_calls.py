"""Fixture: functions and methods that call each other - for call-graph extraction tests."""


def compute_score(data):
    return sum(data)


def normalize(value):
    return value / 100.0


def run_analysis(data):
    score = compute_score(data)
    return normalize(score)


class Analyzer:
    def process(self, data):
        return run_analysis(data)

    def score(self, data):
        return compute_score(data)

    def full_pipeline(self, data):
        raw = self.score(data)
        return normalize(raw)
