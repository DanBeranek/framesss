from __future__ import annotations

from collections.abc import Hashable
from typing import TYPE_CHECKING
from typing import Any
from typing import TypeVar

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

    T = TypeVar("T")


def assemble_subarray_at_indices(
    array: npt.NDArray[np.float64],
    subarray: npt.NDArray[np.float64],
    indices: list[int],
) -> None:
    """
    Insert a subarray into a specified position within a larger array, identified by indices.

    This method modifies the larger array in-place, adding the values from the subarray to the
    elements of the array at the specified indices.

    :var array: The larger array to which the subarray will be added. This array is modified in-place.
    :var subarray: A smaller (n x n) array whose values are to be inserted into the larger array.
    :var indices: A list of integer indices specifying the rows and columns in the larger array
                  where the subarray's values should be added. The indices correspond to the positions
                  in the larger array.

    :return: None. The operation modifies the 'array' argument in-place.

    **Example**::

        large_array = np.zeros((4, 4))
        small_array = np.array([[1, 2], [3, 4]])
        indices = [1, 2]
        assemble_subarray_at_indices(large_array, small_array, indices)
        print(large_array)
        # Output:
        # [[0. 0. 0. 0.]
        #  [0. 1. 2. 0.]
        #  [0. 3. 4. 0.]
        #  [0. 0. 0. 0.]]
    """
    # Split the indices into row and column lists
    rows, cols = zip(*[(i, j) for i in indices for j in indices])
    # Assemble values using indexing
    array[rows, cols] += subarray.flatten()


class DictProxy:
    """
    A proxy class for managing access to a dictionary attribute of another object.

    This class is designed to provide a controlled interface for dictionary
    operations, allowing for additional logic to be implemented when accessing,
    setting, or deleting items in the dictionary.

    :param owner: The object that owns the dictionary this proxy is managing.
    :param dict_name: The name of the dictionary attribute within the owner object.
    """

    def __init__(self, owner: Any, dict_name: str) -> None:
        """
        Init the DictProxy with an owner object and the name of the dictionary attribute.

        :param owner: The object that owns the dictionary.
        :param dict_name: The name of the dictionary attribute in the owner object.
        """
        self.owner = owner
        self.dict_name = dict_name

    def __setitem__(self, key: Hashable, value: Any) -> None:
        """
        Set the value of a key in the proxied dictionary.

        :param key: The key in the dictionary where the value should be set.
        :param value: The value to set for the given key.
        """
        dict_attr = getattr(self.owner, self.dict_name)
        if dict_attr is not None:
            dict_attr[key] = value
        else:
            # Handle None case, e.g., by raising a custom exception or logging a warning
            pass  # Or replace with a more suitable action

    def __getitem__(self, key: Hashable) -> Any:
        """
        Retrieve the value for a given key from the proxied dictionary.

        :param key: The key whose value is to be retrieved.
        :return: The value associated with the given key.
        :raises KeyError: If the dictionary is None or the key is not found.
        """
        dict_attr = getattr(self.owner, self.dict_name)
        if dict_attr is not None:
            return dict_attr.get(
                key
            )  # Using .get() to avoid KeyError if the key doesn't exist
        else:
            # Handle None case
            # pass
            raise KeyError(key)

    def __delitem__(self, key: Hashable) -> None:
        """
        Delete a key-value pair from the proxied dictionary.

        :param key: The key to delete from the dictionary.
        """
        dict_attr = getattr(self.owner, self.dict_name)
        if dict_attr is not None:
            if key in dict_attr:
                del dict_attr[key]
            else:
                # Optionally handle the case where the key doesn't exist
                pass
        else:
            # Handle None case
            pass  # Or replace with a more suitable action

    def __repr__(self) -> str:
        """Return a string representation of the proxied dictionary."""
        dict_attr = getattr(self.owner, self.dict_name)
        return repr(dict_attr) if dict_attr is not None else "None"

    def get(self, key: Hashable, default: T | None = None) -> T | Any:
        """
        Return the value for a given key from the proxied dictionary.

        A default value is returned if the key is not found.

        :param key: The key of the item to retrieve.
        :param default: The default value to return if the key is not found.
        :return: The value associated with the key, or the default value.
        """
        dict_attr = getattr(self.owner, self.dict_name)
        return dict_attr.get(key, default) if dict_attr is not None else default
