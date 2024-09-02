import torch

class HLP:
    """
    A class to handle Hierarchical Label Propagation (HLP) using a lookup table.

    Attributes:
        HLP_lookup_table (torch.Tensor): A square matrix used for label propagation.
    """

    def __init__(self, HLP_lookup_table: torch.Tensor):
        """
        Initializes the HLP class with a given lookup table.

        Args:
            HLP_lookup_table (torch.Tensor): A square matrix used for label propagation.
        """
        self.HLP_lookup_table = torch.Tensor()
        self.update_lookup_table(HLP_lookup_table)

    def __len__(self) -> int:
        """
        Returns the number of rows (or columns) in the lookup table.

        Returns:
            int: The size of the lookup table.
        """
        return len(self.HLP_lookup_table)

    def __repr__(self) -> str:
        """
        Returns a string representation of the HLP object.

        Returns:
            str: A string representation of the HLP object.
        """
        return f'HLP(self.lookup_table.shape={tuple(self.HLP_lookup_table.shape)})'

    def __str__(self) -> str:
        """
        Returns a string representation of the HLP object.

        Returns:
            str: A string representation of the HLP object.
        """
        return self.__repr__()

    def update_lookup_table(self, HLP_lookup_table: torch.Tensor) -> None:
        """
        Updates the lookup table with a new square matrix.

        Args:
            HLP_lookup_table (torch.Tensor): A new square matrix for the lookup table.

        Raises:
            AssertionError: If the provided lookup table is not square.
        """
        assert HLP_lookup_table.shape[0] == HLP_lookup_table.shape[1], f'The lookup table is not a square matrix: lookup_table.shape[0] != lookup_table.shape[1] ({HLP_lookup_table.shape[0]} != {HLP_lookup_table.shape[1]})'
        self.HLP_lookup_table = HLP_lookup_table

    def get_HLP_lookup_table(self) -> torch.Tensor:
        """
        Returns the current lookup table.

        Returns:
            torch.Tensor: The current lookup table.
        """
        return self.HLP_lookup_table

    @property
    def shape(self) -> torch.Size:
        """
        Returns the shape of the HLP lookup table.

        Returns:
            torch.Size: The shape of the HLP lookup table.
        """
        return self.HLP_lookup_table.shape

    def propagate(self, targets: torch.Tensor) -> torch.Tensor:
        """
        Applies Hierarchical Label Propagation (HLP) to the target labels.

        Args:
            targets (torch.Tensor): A tensor of target labels. Can either be a 1D tensor of shape (# of classes) or a 2D tensor of shape (batch_size, # of classes).

        Returns:
            torch.Tensor: The targets with HLP applied.

        Raises:
            ValueError: If the targets tensor does not have 1 or 2 dimensions.
        """
        targets_HLP = targets.clone()

        if len(targets.shape) not in [1, 2]:
            raise ValueError(f'Expected targets to have 1 or 2 dimensions, but got {len(targets_HLP.shape)} dimensions.')

        if len(targets.shape) == 1:
            targets_HLP = targets_HLP.unsqueeze(0)

        # Here, targets_HLP.shape = (batch_size, # of classes)

        # Reshape batch_of_labels to (batch_size, # of classes, 1)
        targets_HLP = targets_HLP.unsqueeze(-1)

        # Multiply with correction_lookup and take max along the appropriate dimension
        targets_HLP = (targets_HLP * self.HLP_lookup_table).max(dim=1).values

        # Set the shape of the output tensor to match the input tensor
        if len(targets.shape) == 1:
            targets_HLP = targets_HLP.squeeze(0)
        return targets_HLP


if __name__ == '__main__':
    HLP_lookup_table = torch.tensor([
        [1, 0, 0],
        [1, 1, 0],
        [0, 0, 1]
    ])
    hlp = HLP(HLP_lookup_table)
    print(hlp)
    print(hlp.shape)
    print("-"*50)
    print(hlp.propagate(torch.tensor(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])))
    # Expected output:
        # tensor([
            # [1, 0, 0],
            # [1, 1, 0],
            # [0, 0, 1]
        # ])
    print("-"*50)
    print(hlp.propagate(torch.tensor([0, 0, 0])))
    # Expected output:
        # tensor([0, 0, 0])
    print("-"*50)
    print(hlp.propagate(torch.tensor([0, 0, 1])))
    # Expected output:
        # tensor([0, 0, 1])
    print("-"*50)
    print(hlp.propagate(torch.tensor([0, 1, 0])))
    # Expected output:
        # tensor([1, 1, 0])
    print("-"*50)
    print(hlp.propagate(torch.tensor([0, 1, 1])))
    # Expected output:
        # tensor([1, 1, 1])
    print("-"*50)
    print(hlp.propagate(torch.tensor([1, 0, 0])))
    # Expected output:
        # tensor([1, 0, 0])
    print("-"*50)
    print(hlp.propagate(torch.tensor([1, 0, 1])))
    # Expected output:
        # tensor([1, 0, 1])
    print("-"*50)
    print(hlp.propagate(torch.tensor([1, 1, 0])))
    # Expected output:
        # tensor([1, 1, 0])
    print("-"*50)
    print(hlp.propagate(torch.tensor([1, 1, 1])))
    # Expected output:
        # tensor([1, 1, 1])
