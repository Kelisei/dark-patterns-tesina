from abc import ABC, abstractmethod


class DarkStrategy(ABC):

    @abstractmethod
    def predict(self,text: str):
        pass

    @abstractmethod
    def predict_multiple(self, texts: list):
        pass


