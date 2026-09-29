"""Actionable exception types for optional real-life benchmark assets."""

from __future__ import annotations


class RealLifeBenchmarkError(RuntimeError):
    """Base class for errors at a real-life benchmark boundary."""


class MissingDependencyError(RealLifeBenchmarkError):
    def __init__(
        self,
        task: str,
        package: str,
        install_hint: str,
        *,
        expected_version: str | None = None,
        required_asset: str | None = None,
        expected_path: object | None = None,
        environment_variable: str | None = None,
        original_error: BaseException | None = None,
    ) -> None:
        version_text = (
            f" Expected/reference version: {expected_version}." if expected_version else ""
        )
        original_text = (
            f" Original error: {type(original_error).__name__}: {original_error}"
            if original_error is not None
            else ""
        )
        asset_text = ""
        if required_asset is not None:
            asset_text = f" Required benchmark asset: {required_asset!r}."
        if expected_path is not None:
            asset_text += f" Expected path: {expected_path!s}."
        if environment_variable is not None:
            asset_text += f" Set {environment_variable} to a valid local path."
        super().__init__(
            f"{task}: optional package {package!r} is unavailable. "
            f"Install/configure it with: {install_hint}.{version_text}"
            f"{asset_text}{original_text} No synthetic/mock fallback is used."
        )
        self.task = task
        self.package = package
        self.expected_version = expected_version


class MissingDataError(RealLifeBenchmarkError):
    def __init__(
        self,
        task: str,
        required_asset: str,
        expected_path: object,
        environment_variable: str,
        *,
        detail: str | None = None,
    ) -> None:
        detail_text = f" {detail}" if detail else ""
        super().__init__(
            f"{task}: required benchmark asset {required_asset!r} was not found. "
            f"Expected path: {expected_path!s}. Set {environment_variable} to a valid "
            f"local path.{detail_text} No synthetic/mock fallback is used."
        )
        self.task = task
        self.required_asset = required_asset
        self.expected_path = str(expected_path)
        self.environment_variable = environment_variable


class BackendCompatibilityError(RealLifeBenchmarkError):
    """The dependency exists, but its API or physical task shape is incompatible."""


__all__ = [
    "BackendCompatibilityError",
    "MissingDataError",
    "MissingDependencyError",
    "RealLifeBenchmarkError",
]
