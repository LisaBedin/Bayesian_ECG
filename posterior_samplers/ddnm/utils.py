import torch


def get_special_methods(A_funcs, type_problem: str):
    """Get ``Lambda`` and ``Lambda_noise`` methods needed in DDNM."""

    pb = SUPPORTED_PROBELEMS.get(type_problem, None)

    if pb is None:
        raise ValueError("Unsupported type of problem")

    Lambda_func = lambda *args: pb.Lambda(A_funcs, *args)
    Lambda_noise_func = lambda *args: pb.Lambda_noise(A_funcs, *args)

    return Lambda_func, Lambda_noise_func


# --- Code copy/pasted then adapted from
# https://github.com/wyhuai/DDNM/blob/main/functions/svd_operators.py


# Inpainting
class Inpainting:

    def Lambda(self, vec, a, sigma_y, sigma_t, eta):

        temp = (
            vec.clone()
            .reshape(vec.shape[0], self.channels, -1)
            .permute(0, 2, 1)
            .reshape(vec.shape[0], -1)
        )
        out = torch.zeros_like(temp)
        out[:, : self.kept_indices.shape[0]] = temp[:, self.kept_indices]
        out[:, self.kept_indices.shape[0] :] = temp[:, self.missing_indices]

        singulars = self._singulars
        lambda_t = torch.ones(temp.size(1), device=vec.device)
        temp_singulars = torch.zeros(temp.size(1), device=vec.device)
        temp_singulars[: singulars.size(0)] = singulars
        singulars = temp_singulars
        inverse_singulars = 1.0 / singulars
        inverse_singulars[singulars == 0] = 0.0
        if type(sigma_y) == torch.Tensor:
            sigma_bool = float(sigma_y.max()) > 0
            #if sigma_bool:
            #    change_index = ((sigma_t < a * sigma_y.view(-1, 1) * inverse_singulars.view(sigma_y.shape[0], -1)) * 1.0).flatten()
            #    lambda_t = lambda_t * (-change_index + 1.0) + change_index * (
            #        singulars.view(sigma_y.shape[0], -1) * sigma_t * (1 - eta ** 2) ** 0.5 / a / sigma_y.view(-1, 1)
            #    ).flatten()
        else:
            sigma_bool = sigma_y != 0

        if a != 0 and sigma_bool:
            change_index = (sigma_t < a * sigma_y * inverse_singulars) * 1.0
            lambda_t = lambda_t * (-change_index + 1.0) + change_index * (
            singulars * sigma_t * (1 - eta**2) ** 0.5 / a / sigma_y
        )

        lambda_t = lambda_t.reshape(1, -1)
        out = out * lambda_t

        result = torch.zeros_like(temp)
        result[:, self.kept_indices] = out[:, : self.kept_indices.shape[0]]
        result[:, self.missing_indices] = out[:, self.kept_indices.shape[0] :]
        return (
            result.reshape(vec.shape[0], -1, self.channels)
            .permute(0, 2, 1)
            .reshape(vec.shape[0], -1)
        )

    def Lambda_noise(self, vec, a, sigma_y, sigma_t, eta, epsilon):
        temp_vec = (
            vec.clone()
            .reshape(vec.shape[0], self.channels, -1)
            .permute(0, 2, 1)
            .reshape(vec.shape[0], -1)
        )
        out_vec = torch.zeros_like(temp_vec)
        out_vec[:, : self.kept_indices.shape[0]] = temp_vec[:, self.kept_indices]
        out_vec[:, self.kept_indices.shape[0] :] = temp_vec[:, self.missing_indices]

        temp_eps = (
            epsilon.clone()
            .reshape(vec.shape[0], self.channels, -1)
            .permute(0, 2, 1)
            .reshape(vec.shape[0], -1)
        )
        out_eps = torch.zeros_like(temp_eps)
        out_eps[:, : self.kept_indices.shape[0]] = temp_eps[:, self.kept_indices]
        out_eps[:, self.kept_indices.shape[0] :] = temp_eps[:, self.missing_indices]

        singulars = self._singulars
        d1_t = torch.ones(temp_vec.size(1), device=vec.device) * sigma_t * eta
        d2_t = (
            torch.ones(temp_vec.size(1), device=vec.device)
            * sigma_t
            * (1 - eta**2) ** 0.5
        )

        temp_singulars = torch.zeros(temp_vec.size(1), device=vec.device)
        temp_singulars[: singulars.size(0)] = singulars
        singulars = temp_singulars
        inverse_singulars = 1.0 / singulars
        inverse_singulars[singulars == 0] = 0.0
        if type(sigma_y) == torch.Tensor:
            sigma_bool = float(sigma_y.max()) > 0
        else:
            sigma_bool = sigma_y != 0

        if a != 0 and sigma_bool:
            change_index = (sigma_t < a * sigma_y * inverse_singulars) * 1.0
            d1_t = d1_t * (-change_index + 1.0) + change_index * sigma_t * eta
            d2_t = d2_t * (-change_index + 1.0)

            change_index = (sigma_t > a * sigma_y * inverse_singulars) * 1.0
            d1_t = d1_t * (-change_index + 1.0) + torch.sqrt(
                change_index * (sigma_t**2 - a**2 * sigma_y**2 * inverse_singulars**2)
            )
            d2_t = d2_t * (-change_index + 1.0)

            change_index = (singulars == 0) * 1.0
            d1_t = d1_t * (-change_index + 1.0) + change_index * sigma_t * eta
            d2_t = (
                d2_t * (-change_index + 1.0)
                + change_index * sigma_t * (1 - eta**2) ** 0.5
            )

        d1_t = d1_t.reshape(1, -1)
        d2_t = d2_t.reshape(1, -1)
        out_vec = out_vec * d1_t
        out_eps = out_eps * d2_t

        result_vec = torch.zeros_like(temp_vec)
        result_vec[:, self.kept_indices] = out_vec[:, : self.kept_indices.shape[0]]
        result_vec[:, self.missing_indices] = out_vec[:, self.kept_indices.shape[0] :]
        result_vec = (
            result_vec.reshape(vec.shape[0], -1, self.channels)
            .permute(0, 2, 1)
            .reshape(vec.shape[0], -1)
        )

        result_eps = torch.zeros_like(temp_eps)
        result_eps[:, self.kept_indices] = out_eps[:, : self.kept_indices.shape[0]]
        result_eps[:, self.missing_indices] = out_eps[:, self.kept_indices.shape[0] :]
        result_eps = (
            result_eps.reshape(vec.shape[0], -1, self.channels)
            .permute(0, 2, 1)
            .reshape(vec.shape[0], -1)
        )

        return result_vec + result_eps


class SuperResolution:

    def Lambda(self, vec, a, sigma_y, sigma_t, eta):
        singulars = self.singulars_small

        patches = vec.clone().reshape(
            vec.shape[0], self.channels, self.img_dim, self.img_dim
        )
        patches = patches.unfold(2, self.ratio, self.ratio).unfold(
            3, self.ratio, self.ratio
        )
        patches = patches.contiguous().reshape(
            vec.shape[0], self.channels, -1, self.ratio**2
        )

        patches = torch.matmul(
            self.Vt_small, patches.reshape(-1, self.ratio**2, 1)
        ).reshape(vec.shape[0], self.channels, -1, self.ratio**2)

        lambda_t = torch.ones(self.ratio**2, device=vec.device)

        temp = torch.zeros(self.ratio**2, device=vec.device)
        temp[: singulars.size(0)] = singulars
        singulars = temp
        inverse_singulars = 1.0 / singulars
        inverse_singulars[singulars == 0] = 0.0

        if type(sigma_y) == torch.Tensor:
            sigma_bool = float(sigma_y.max()) > 0
        else:
            sigma_bool = sigma_y != 0

        if a != 0 and sigma_bool:
            change_index = (sigma_t < a * sigma_y * inverse_singulars) * 1.0
            lambda_t = lambda_t * (-change_index + 1.0) + change_index * (
                singulars * sigma_t * (1 - eta**2) ** 0.5 / a / sigma_y
            )

        lambda_t = lambda_t.reshape(1, 1, 1, -1)

        patches = patches * lambda_t

        patches = torch.matmul(self.V_small, patches.reshape(-1, self.ratio**2, 1))

        patches = patches.reshape(
            vec.shape[0], self.channels, self.y_dim, self.y_dim, self.ratio, self.ratio
        )
        patches = patches.permute(0, 1, 2, 4, 3, 5).contiguous()
        patches = patches.reshape(vec.shape[0], self.channels * self.img_dim**2)

        return patches

    def Lambda_noise(self, vec, a, sigma_y, sigma_t, eta, epsilon):
        singulars = self.singulars_small

        patches_vec = vec.clone().reshape(
            vec.shape[0], self.channels, self.img_dim, self.img_dim
        )
        patches_vec = patches_vec.unfold(2, self.ratio, self.ratio).unfold(
            3, self.ratio, self.ratio
        )
        patches_vec = patches_vec.contiguous().reshape(
            vec.shape[0], self.channels, -1, self.ratio**2
        )

        patches_eps = epsilon.clone().reshape(
            vec.shape[0], self.channels, self.img_dim, self.img_dim
        )
        patches_eps = patches_eps.unfold(2, self.ratio, self.ratio).unfold(
            3, self.ratio, self.ratio
        )
        patches_eps = patches_eps.contiguous().reshape(
            vec.shape[0], self.channels, -1, self.ratio**2
        )

        d1_t = torch.ones(self.ratio**2, device=vec.device) * sigma_t * eta
        d2_t = (
            torch.ones(self.ratio**2, device=vec.device) * sigma_t * (1 - eta**2) ** 0.5
        )

        temp = torch.zeros(self.ratio**2, device=vec.device)
        temp[: singulars.size(0)] = singulars
        singulars = temp
        inverse_singulars = 1.0 / singulars
        inverse_singulars[singulars == 0] = 0.0

        if type(sigma_y) == torch.Tensor:
            sigma_bool = float(sigma_y.max()) > 0
        else:
            sigma_bool = sigma_y != 0

        if a != 0 and sigma_bool:
            change_index = (sigma_t < a * sigma_y * inverse_singulars) * 1.0
            d1_t = d1_t * (-change_index + 1.0) + change_index * sigma_t * eta
            d2_t = d2_t * (-change_index + 1.0)

            change_index = (sigma_t > a * sigma_y * inverse_singulars) * 1.0
            d1_t = d1_t * (-change_index + 1.0) + torch.sqrt(
                change_index * (sigma_t**2 - a**2 * sigma_y**2 * inverse_singulars**2)
            )
            d2_t = d2_t * (-change_index + 1.0)

            change_index = (singulars == 0) * 1.0
            d1_t = d1_t * (-change_index + 1.0) + change_index * sigma_t * eta
            d2_t = (
                d2_t * (-change_index + 1.0)
                + change_index * sigma_t * (1 - eta**2) ** 0.5
            )

        d1_t = d1_t.reshape(1, 1, 1, -1)
        d2_t = d2_t.reshape(1, 1, 1, -1)
        patches_vec = patches_vec * d1_t
        patches_eps = patches_eps * d2_t

        patches_vec = torch.matmul(
            self.V_small, patches_vec.reshape(-1, self.ratio**2, 1)
        )

        patches_vec = patches_vec.reshape(
            vec.shape[0], self.channels, self.y_dim, self.y_dim, self.ratio, self.ratio
        )
        patches_vec = patches_vec.permute(0, 1, 2, 4, 3, 5).contiguous()
        patches_vec = patches_vec.reshape(vec.shape[0], self.channels * self.img_dim**2)

        patches_eps = torch.matmul(
            self.V_small, patches_eps.reshape(-1, self.ratio**2, 1)
        )

        patches_eps = patches_eps.reshape(
            vec.shape[0], self.channels, self.y_dim, self.y_dim, self.ratio, self.ratio
        )
        patches_eps = patches_eps.permute(0, 1, 2, 4, 3, 5).contiguous()
        patches_eps = patches_eps.reshape(vec.shape[0], self.channels * self.img_dim**2)

        return patches_vec + patches_eps


# ---

SUPPORTED_PROBELEMS = {"inpainting": Inpainting, "sr": SuperResolution}
