
import numpy as np
from scipy.stats import norm

from .abstract_option_model import AbstractOptionModel


class AbstractDisplacedDiffusionModel(AbstractOptionModel):
   

    def __init__(
        self, S: float, K: float, r: float, sigma: float, T: float, beta: float
    ):
        self.S = S
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T
        self.beta = beta

        self.F = S * np.exp(r * T)
        self.adjusted_F = self.F / self.beta
        self.adjusted_K = self.K + ((1 - self.beta) / self.beta) * self.F
        self.adjusted_sigma = self.sigma * self.beta
        self.discount_factor = np.exp(-self.r * self.T)

        self.d1 = self._calculate_d1()
        self.d2 = self._calculate_d2()

    def _calculate_d1(self) -> float:
        return (
            np.log(self.adjusted_F / self.adjusted_K)
            + 0.5 * self.adjusted_sigma**2 * self.T
        ) / (self.adjusted_sigma * np.sqrt(self.T))

    def _calculate_d2(self) -> float:
        return self.d1 - self.adjusted_sigma * np.sqrt(self.T)


class VanillaDisplacedDiffusionModel(AbstractDisplacedDiffusionModel):
    def calculate_call_price(self) -> float:
        return self.discount_factor * (
            self.adjusted_F * norm.cdf(self.d1) - self.adjusted_K * norm.cdf(self.d2)
        )

    def calculate_put_price(self) -> float:
        return self.discount_factor * (
            self.adjusted_K * norm.cdf(-self.d2) - self.adjusted_F * norm.cdf(-self.d1)
        )


class DigitalCashOrNothingDisplacedDiffusionModel(AbstractDisplacedDiffusionModel):
    def calculate_call_price(self) -> float:
        return self.discount_factor * norm.cdf(self.d2)

    def calculate_put_price(self) -> float:
        return self.discount_factor * norm.cdf(-self.d2)


class DigitalAssetOrNothingDisplacedDiffusionModel(AbstractDisplacedDiffusionModel):
    def calculate_call_price(self) -> float:
        return self.discount_factor * self.adjusted_F * norm.cdf(self.d1)

    def calculate_put_price(self) -> float:
        return self.discount_factor * self.adjusted_F * norm.cdf(-self.d1)
