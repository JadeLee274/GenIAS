from .common_import import *


class pretextloss(nn.Module):
    """
    Loss function for pretext stage of CARLA.
    
    Parameters:
        batch_size:     Batch size.
        temperature:    The cardinality of the set of all triplets (a, p, n)
        initial_margin: Initial margin that controlls the minimum distance
                        between positive and negative pairs.
        adjust_factor:  Adjustment factor when updating the margin.

    Optimizing this loss is for decreasing the distance between the anchor and
    its corresponding positive sample, while simultaneously increasing the
    distance between the anchor and its corresponding negative sample, in the
    representation space.

    Such approach encourages the model to learn a representation that can
    differentiate between normal and abnormal windows.
    """
    def __init__(
        self,
        batch_size: int,
        temperature: int,
        initial_margin: float = 1.0,
        adjust_factor: float = 0.1,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.batch_size = batch_size
        self.margin = initial_margin
        self.adjust_factor = adjust_factor

    def forward(
        self,
        representations: Tensor,
        current_loss: Optional[float] = None,
    ) -> Tensor:
        anchor, positive_pair, negative_pair = torch.split(
            tensor=representations,
            split_size_or_sections=self.batch_size,
            dim=0,
        )
        anchor = F.normalize(anchor, dim=-1)
        positive_pair = F.normalize(positive_pair, dim=-1)
        negative_pair = F.normalize(negative_pair, dim=-1)

        # update margin
        if current_loss is not None:
            self.margin = max(
                0.01,
                self.margin - self.adjust_factor * current_loss
            )
        
        positive_dist = torch.sum(
            input=(anchor - positive_pair) ** 2,
            dim=-1
        ) / self.temperature

        negative_dist = torch.sum(
            input=torch.pow(anchor.unsqueeze(1) - negative_pair),
            dim=-1
        ) / self.temperature

        hard_negetive_dist = torch.min(
            input=negative_dist,
            dim=-1,
        )[0]

        loss = torch.clamp(
            input=self.margin + positive_dist - hard_negetive_dist,
            min=0.0,
        )
        loss = torch.mean(loss)

        return loss
    
    def consine_similarity(
        self,
        x1: Tensor,
        x2: Tensor,
    ) -> Tensor:
        dot_prod = torch.sum(x1 * x2, dim=1)
        norm_prod = torch.norm(x1, dim=1) * torch.norm(x2, dim=1)
        cos_similarity = dot_prod / norm_prod
        return cos_similarity
    
    def euclidean_dist(
        self,
        x1: Tensor,
        x2: Tensor,
    ) -> Tensor:
        return torch.sqrt(((x1 - x2) ** 2).sum(dim=1))
    

class classificationloss(nn.Module):
    """
    Classification loss for CARLA's self-supervised classification stage.

    Parameters:
        entropy_weight:       Weight for entropy loss. Default 2.0.
        inconsistency_weight: Weight for inconsistency loss. Default is 1.0.

    Optimizing this loss is for classifying the normal data to particular
    class. 
    """
    def __init__(
        self,
        entropy_weight: float = 2.0,
        inconsistency_weight: float = 1.0,
    ) -> None:
        self.softmax = nn.Softmax(dim=1)
        self.bceloss = nn.BCELoss()
        self.entropy_weight = entropy_weight
        self.inconsistency_weight = inconsistency_weight

    def forward(
        self,
        anchors: Tensor,
        nearest_neighbors: Tensor,
        furthest_neighbors: Tensor,
    ) -> Tensor:
        B, N = anchors.shape
        anchors_probability = self.softmax(anchors)
        positives_pobability = self.softmax(nearest_neighbors)
        negative_probability = self.softmax(furthest_neighbors)

        similarity = torch.bmm(
            anchors_probability.view(B, 1, N),
            positives_pobability.view(B, N, 1),
        ).squeeze() # (B, 1)
        positive_pseudolabel = torch.ones_like(similarity) # (B, 1)
        consistency_loss = self.bceloss(
            similarity,
            positive_pseudolabel
        )

        negative_similarity = torch.bmm(
            anchors_probability.view(B, 1, N),
            negative_probability.view(B, N, 1),
        ).squeeze()
        negative_pseudolabel = torch.zeros_like(negative_similarity) # (B, 1)
        inconsistency_loss = self.bceloss(
            negative_similarity,
            negative_pseudolabel
        )

        entropy_loss = self.entropy(
            x=torch.mean(anchors_probability, dim=0),
            input_as_probability=True,
        )

        total_loss = consistency_loss \
                    - self.entropy_weight * entropy_loss \
                    - self.inconsistency_weight * inconsistency_loss
        
        return total_loss, consistency_loss, inconsistency_loss, entropy_loss
    
    def entropy(
        self,
        x: Tensor,
        input_as_probability: bool,
    ) -> Tensor:
        if input_as_probability:
            x_ = torch.clamp(x, min=1e-8)
            b = x_ * torch.log(x_)
        else:
            b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        
        if len(b.size()) == 2:
            return -b.sum(dim=1).mean()
        elif len(b.size()) == 1:
            return -b.sum()
        else:
            raise ValueError(f'Input tensor is {b.size()}-dimensional.')

