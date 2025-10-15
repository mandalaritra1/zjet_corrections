import numpy
import coffea.processor
import coffea.util

class WeightStatistics:
    def __init__(self, sumw=0.0, sumw2=0.0, minw=numpy.inf, maxw=-numpy.inf, n=0):
        self.sumw = sumw
        self.sumw2 = sumw2
        self.minw = minw
        self.maxw = maxw
        self.n = n

    def __repr__(self):
        return f"WeightStatistics(sumw={self.sumw}, sumw2={self.sumw2}, minw={self.minw}, maxw={self.maxw}, n={self.n})"

    def identity(self):
        return WeightStatistics()

    def add(self, other):
        self.sumw += other.sumw
        self.sumw2 += other.sumw2
        self.minw = min(self.minw, other.minw)
        self.maxw = max(self.maxw, other.maxw)
        self.n += other.n

    def __add__(self, other):
        temp = WeightStatistics(self.sumw, self.sumw2, self.minw, self.maxw, self.n)
        return temp.add(other)

    def __iadd__(self, other):
        return self.add(other)
        
class Weights:
    """Container for event weights and associated systematic shifts

    This container keeps track of correction factors and systematic
    effects that can be encoded as multiplicative modifiers to the event weight.
    All weights are stored in vector form.

    Parameters
    ----------
        size : int
            size of the weight arrays to be handled (i.e. the number of events / instances).
        storeIndividual : bool, optional
            store not only the total weight + variations, but also each individual weight.
            Default is false.
    """

    def __init__(self, size, storeIndividual=False):
        self._weight = numpy.ones(size)
        self._weights = {}
        self._modifiers = {}
        self._weightStats = coffea.processor.dict_accumulator()
        self._storeIndividual = storeIndividual

    @property
    def weightStatistics(self):
        return self._weightStats

    def add(self, name, weight, weightUp=None, weightDown=None, shift=False):
        """Add a new weight

        Adds a named correction to the event weight, and optionally also associated
        systematic uncertainties.

        Parameters
        ----------
            name : str
                name of correction
            weight : numpy.ndarray
                the nominal event weight associated with the correction
            weightUp : numpy.ndarray, optional
                weight with correction uncertainty shifted up (if available)
            weightDown : numpy.ndarray, optional
                weight with correction uncertainty shifted down. If ``weightUp`` is supplied, and
                the correction uncertainty is symmetric, this can be set to None to auto-calculate
                the down shift as ``1 / weightUp``.
            shift : bool, optional
                if True, interpret weightUp and weightDown as a realtive difference (additive) to the
                nominal value

        .. note:: ``weightUp`` and ``weightDown`` are assumed to be rvalue-like and may be modified in-place by this function
        """
        if name.endswith("Up") or name.endswith("Down"):
            raise ValueError(
                "Avoid using 'Up' and 'Down' in weight names, instead pass appropriate shifts to add() call"
            )
        weight = coffea.util._ensure_flat(weight, allow_missing=True)
        if isinstance(weight, numpy.ma.MaskedArray):
            # TODO what to do with option-type? is it representative of unknown weight
            # and we default to one or is it an invalid weight and we should never use this
            # event in the first place (0) ?
            weight = weight.filled(1.0)
        self._weight = self._weight * weight
        if self._storeIndividual:
            self._weights[name] = weight
        self.__add_variation(name, weight, weightUp, weightDown, shift)
        self._weightStats[name] = WeightStatistics(
            weight.sum(),
            (weight**2).sum(),
            weight.min(),
            weight.max(),
            weight.size,
        )

    def add_multivariation(
        self, name, weight, modifierNames, weightsUp, weightsDown, shift=False
    ):
        """Add a new weight with multiple variations

        Each variation of a single weight is given a different modifier name.
        This is particularly useful e.g. for btag SF variations.

        Parameters
        ----------
            name : str
                name of correction
            weight : numpy.ndarray
                the nominal event weight associated with the correction
            modifierNames: list of str
                list of modifiers for each set of weights variation
            weightsUp : list of numpy.ndarray
                weight with correction uncertainty shifted up (if available)
            weightsDown : list of numpy.ndarray
                weight with correction uncertainty shifted down. If ``weightUp`` is supplied, and
                the correction uncertainty is symmetric, this can be set to None to auto-calculate
                the down shift as ``1 / weightUp``.
            shift : bool, optional
                if True, interpret weightUp and weightDown as a realtive difference (additive) to the
                nominal value

        .. note:: ``weightUp`` and ``weightDown`` are assumed to be rvalue-like and may be modified in-place by this function
        """
        if name.endswith("Up") or name.endswith("Down"):
            raise ValueError(
                "Avoid using 'Up' and 'Down' in weight names, instead pass appropriate shifts to add() call"
            )
        weight = coffea.util._ensure_flat(weight, allow_missing=True)
        if isinstance(weight, numpy.ma.MaskedArray):
            # TODO what to do with option-type? is it representative of unknown weight
            # and we default to one or is it an invalid weight and we should never use this
            # event in the first place (0) ?
            weight = weight.filled(1.0)
        self._weight = self._weight * weight
        if self._storeIndividual:
            self._weights[name] = weight
        # Now loop on the variations
        if len(modifierNames) > 0:
            if len(modifierNames) != len(weightsUp) or len(modifierNames) != len(
                weightsDown
            ):
                raise ValueError(
                    "Provide the same number of modifier names related to the list of modified weights"
                )
        for modifier, weightUp, weightDown in zip(
            modifierNames, weightsUp, weightsDown
        ):
            systName = f"{name}_{modifier}"
            self.__add_variation(systName, weight, weightUp, weightDown, shift)
        self._weightStats[name] = WeightStatistics(
            weight.sum(),
            (weight**2).sum(),
            weight.min(),
            weight.max(),
            weight.size,
        )

    def __add_variation(
        self, name, weight, weightUp=None, weightDown=None, shift=False
    ):
        """Helper function to add a weight variation.

        Parameters
        ----------
            name : str
                name of systematic variation (just the name of the weight if only
                one variation is added, or `name_syst` for multiple variations)
            weight : numpy.ndarray
                the nominal event weight associated with the correction
            weightUp : numpy.ndarray, optional
                weight with correction uncertainty shifted up (if available)
            weightDown : numpy.ndarray, optional
                weight with correction uncertainty shifted down. If ``weightUp`` is supplied, and
                the correction uncertainty is symmetric, this can be set to None to auto-calculate
                the down shift as ``1 / weightUp``.
            shift : bool, optional
                if True, interpret weightUp and weightDown as a realtive difference (additive) to the
                nominal value

        .. note:: ``weightUp`` and ``weightDown`` are assumed to be rvalue-like and may be modified in-place by this function
        """
        if weightUp is not None:
            weightUp = coffea.util._ensure_flat(weightUp, allow_missing=True)
            if isinstance(weightUp, numpy.ma.MaskedArray):
                weightUp = weightUp.filled(1.0)
            if shift:
                weightUp += weight
            weightUp[weight != 0.0] /= weight[weight != 0.0]
            self._modifiers[name + "Up"] = weightUp
        if weightDown is not None:
            weightDown = coffea.util._ensure_flat(weightDown, allow_missing=True)
            if isinstance(weightDown, numpy.ma.MaskedArray):
                weightDown = weightDown.filled(1.0)
            if shift:
                weightDown = weight - weightDown
            weightDown[weight != 0.0] /= weight[weight != 0.0]
            self._modifiers[name + "Down"] = weightDown

    def weight(self, modifier=None):
        """Current event weight vector

        Parameters
        ----------
            modifier : str, optional
                if supplied, provide event weight corresponding to a particular
                systematic uncertainty shift, of form ``str(name + 'Up')`` or (Down)

        Returns
        -------
            weight : numpy.ndarray
                The weight vector, possibly modified by the effect of a given systematic variation.
        """
        if modifier is None:
            return self._weight
        elif "Down" in modifier and modifier not in self._modifiers:
            return self._weight / self._modifiers[modifier.replace("Down", "Up")]
        return self._weight * self._modifiers[modifier]

    def partial_weight(self, include=[], exclude=[], modifier = None):
        """Partial event weight vector

        Return a partial weight by multiplying a subset of all weights.
        Can be operated either by specifying weights to include or
        weights to exclude, but not both at the same time. The method
        can only be used if the individual weights are stored via the
        ``storeIndividual`` argument in the `Weights` initializer.

        Parameters
        ----------
            include : list
                Weight names to include, defaults to []
            exclude : list
                Weight names to exclude, defaults to []
            modifier : str, optional
                if supplied, provide event weight corresponding to a particular
                systematic uncertainty shift, of form ``str(name + 'Up')`` or (Down)
        Returns
        -------
            weight : numpy.ndarray
                The weight vector, corresponding to only the effect of the
                corrections specified.
        """
        if not self._storeIndividual:
            raise ValueError(
                "To be able to request weight exclusion, use storeIndividual=True when creating Weights object."
            )
        if (include and exclude) or not (include or exclude):
            raise ValueError(
                "Need to specify exactly one of the 'exclude' or 'include' arguments."
            )

        names = set(self._weights.keys())
        if include:
            names = names & set(include)
        if exclude:
            names = names - set(exclude)

        w = numpy.ones(self._weight.size)
        for name in names:
            w *= self._weights[name]
        if modifier == None:
            return w
        elif modifier.replace("Down", "").replace("Up", "") not in names:
            raise ValueError(
                f"Modifier {modifier} is not in the list of included weights"
            )
        elif "Down" in modifier and modifier not in self._modifiers:
            return w / self._modifiers[modifier.replace("Down", "Up")]
        return w * self._modifiers[modifier]

    @property
    def variations(self):
        """List of available modifiers"""
        keys = set(self._modifiers.keys())
        # add any missing 'Down' variation
        for k in self._modifiers.keys():
            keys.add(k.replace("Up", "Down"))
        return keys
