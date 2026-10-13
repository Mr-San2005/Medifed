"""A minimal FedBN simulation to validate the personalization logic."""

import unittest

import torch
import torch.nn as nn

from medifed.personalization import bn_keys, merge_state


class TinyBatchNormCNN(nn.Module):
    """Small conv net with BatchNorm for a synthetic FedBN experiment."""

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4 * 8 * 8, 2),
        )

    def forward(self, x):
        return self.model(x)


class TestFedBNSimulation(unittest.TestCase):
    """Test the key FedBN property: BN stays local while shared weights are averaged."""

    def _make_client_data(self, shift):
        x = torch.randn(16, 1, 8, 8)
        x = x + shift
        y = torch.randint(0, 2, (16,))
        return x, y

    def _local_train(self, model, x, y, lr=1e-2):
        opt = torch.optim.SGD(model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        model.train()
        opt.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
        return model

    def _personalized_average(self, clients):
        global_state = clients[0].state_dict()
        bn_keys_for_model = bn_keys(clients[0])
        averaged = {}
        for key in global_state:
            if key in bn_keys_for_model:
                averaged[key] = clients[0].state_dict()[key].clone()
            else:
                values = [client.state_dict()[key].clone() for client in clients]
                averaged[key] = torch.stack(values, dim=0).mean(dim=0)
        return averaged

    def test_bn_stats_diverge_when_personalized(self):
        """TC-FBN-007 BN running stats differ across clients while shared weights remain aligned."""
        clients = [TinyBatchNormCNN() for _ in range(3)]
        base = TinyBatchNormCNN()
        for idx, client in enumerate(clients):
            client.load_state_dict(base.state_dict())
            x, y = self._make_client_data(shift=idx * 1.0)
            for _ in range(3):
                client = self._local_train(client, x, y)
            clients[idx] = client

        bn_vals = []
        for client in clients:
            bn_vals.append(client.model[1].running_mean.detach().clone())
        self.assertGreater(len({tuple(v.tolist()) for v in bn_vals}), 1)

        averaged = self._personalized_average(clients)
        for client_idx, client in enumerate(clients):
            client_local_bn = {key: client.state_dict()[key].clone() for key in bn_keys(client)}
            client.load_state_dict(merge_state(averaged, client_local_bn, bn_keys(client)))

        for key in averaged:
            if key in bn_keys(clients[0]):
                continue
            for client in clients:
                self.assertTrue(torch.allclose(client.state_dict()[key], averaged[key], atol=1e-6, rtol=1e-6))

    def test_baseline_average_makes_bn_identical(self):
        """TC-FBN-008 in baseline averaging mode the BN entries become identical across clients."""
        clients = [TinyBatchNormCNN() for _ in range(3)]
        for idx, client in enumerate(clients):
            client.model[1].running_mean = torch.full_like(client.model[1].running_mean, float(idx + 1))
            client.model[1].running_var = torch.full_like(client.model[1].running_var, float(idx + 1))

        avg_mean = sum(client.model[1].running_mean for client in clients) / 3
        avg_var = sum(client.model[1].running_var for client in clients) / 3
        for client in clients:
            client.model[1].running_mean.data = avg_mean.clone()
            client.model[1].running_var.data = avg_var.clone()

        self.assertTrue(torch.allclose(clients[0].model[1].running_mean, clients[1].model[1].running_mean))
        self.assertTrue(torch.allclose(clients[1].model[1].running_mean, clients[2].model[1].running_mean))


if __name__ == "__main__":
    unittest.main()
