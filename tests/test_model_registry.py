"""Tests for MLflow Model Registry utilities.

This module tests model registration, promotion, loading, and metadata retrieval functionality in
src/models/model_registry.py.
"""

from unittest.mock import Mock, call, patch

from src.models import model_registry


class TestGetMLflowClient:
    """Tests for get_mlflow_client()."""

    @patch("src.models.model_registry.MlflowClient")
    def test_get_mlflow_client_returns_client(self, mock_client_class):
        """Test that get_mlflow_client returns an MlflowClient instance."""
        # Arrange
        mock_client_instance = Mock()
        mock_client_class.return_value = mock_client_instance

        # Act
        client = model_registry.get_mlflow_client()

        # Assert
        mock_client_class.assert_called_once()
        assert client == mock_client_instance


class TestRegisterModel:
    """Tests for register_model()."""

    @patch("src.models.model_registry.log_lightgbm_model")
    @patch("src.models.model_registry.get_mlflow_client")
    def test_register_model_basic(self, mock_get_client, mock_log_model):
        """Logs the LightGBM model and returns the new version."""
        mock_model = Mock()
        mock_log_model.return_value = Mock(registered_model_version=1)

        version = model_registry.register_model(mock_model, model_name="rossmann-model")

        mock_log_model.assert_called_once_with(
            mock_model, name="model", registered_model_name="rossmann-model"
        )
        mock_get_client.return_value.update_model_version.assert_not_called()
        assert version == "1"

    @patch("src.models.model_registry.log_lightgbm_model")
    @patch("src.models.model_registry.get_mlflow_client")
    def test_register_model_with_description(self, mock_get_client, mock_log_model):
        """Sets the version description when one is given."""
        mock_log_model.return_value = Mock(registered_model_version="2")

        version = model_registry.register_model(
            Mock(), model_name="rossmann-model", description="Test description"
        )

        mock_get_client.return_value.update_model_version.assert_called_once_with(
            name="rossmann-model", version="2", description="Test description"
        )
        assert version == "2"


class TestPromoteModel:
    """Tests for promote_model()."""

    @patch("src.models.model_registry.get_mlflow_client")
    def test_promote_model_to_staging(self, mock_get_client):
        """Test promoting a model to Staging stage."""
        # Arrange
        mock_client = Mock()
        mock_client.get_latest_versions.return_value = []  # No existing models
        mock_get_client.return_value = mock_client

        # Act
        model_registry.promote_model(model_name="rossmann-model", version="1", stage="Staging")

        # Assert
        mock_client.get_latest_versions.assert_called_once_with(
            "rossmann-model", stages=["Staging"]
        )
        mock_client.transition_model_version_stage.assert_called_once_with(
            name="rossmann-model", version="1", stage="Staging"
        )

    @patch("src.models.model_registry.get_mlflow_client")
    def test_promote_model_to_production_with_archival(self, mock_get_client):
        """Test promoting to Production archives existing Production model."""
        # Arrange
        mock_client = Mock()

        # Existing Production model
        mock_existing = Mock()
        mock_existing.version = "5"
        mock_client.get_latest_versions.return_value = [mock_existing]
        mock_get_client.return_value = mock_client

        # Act
        model_registry.promote_model(
            model_name="rossmann-model", version="7", stage="Production", archive_existing=True
        )

        # Assert
        # Should archive existing model first
        assert mock_client.transition_model_version_stage.call_count == 2
        calls = mock_client.transition_model_version_stage.call_args_list

        # First call: archive existing
        assert calls[0] == call(name="rossmann-model", version="5", stage="Archived")

        # Second call: promote new model
        assert calls[1] == call(name="rossmann-model", version="7", stage="Production")

    @patch("src.models.model_registry.get_mlflow_client")
    def test_promote_model_without_archival(self, mock_get_client):
        """Test promoting without archiving existing models."""
        # Arrange
        mock_client = Mock()
        mock_existing = Mock()
        mock_existing.version = "3"
        mock_client.get_latest_versions.return_value = [mock_existing]
        mock_get_client.return_value = mock_client

        # Act
        model_registry.promote_model(
            model_name="rossmann-model",
            version="4",
            stage="Production",
            archive_existing=False,
        )

        # Assert
        # Should only promote, not archive
        mock_client.transition_model_version_stage.assert_called_once_with(
            name="rossmann-model", version="4", stage="Production"
        )

    @patch("src.models.model_registry.get_mlflow_client")
    def test_promote_model_to_archived(self, mock_get_client):
        """Test promoting directly to Archived stage."""
        # Arrange
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        # Act
        model_registry.promote_model(model_name="rossmann-model", version="2", stage="Archived")

        # Assert
        # Should not try to archive existing Archived models
        mock_client.get_latest_versions.assert_not_called()
        mock_client.transition_model_version_stage.assert_called_once_with(
            name="rossmann-model", version="2", stage="Archived"
        )


class TestGetModelVersion:
    """Tests for get_model_version()."""

    @patch("src.models.model_registry.get_mlflow_client")
    def test_get_model_version_production(self, mock_get_client, mock_model_version):
        """Test retrieving Production model version."""
        # Arrange
        mock_client = Mock()
        mock_model_version.current_stage = "Production"
        mock_model_version.version = "5"
        mock_client.get_latest_versions.return_value = [mock_model_version]
        mock_get_client.return_value = mock_client

        # Act
        version = model_registry.get_model_version("rossmann-model", stage="Production")

        # Assert
        mock_client.get_latest_versions.assert_called_once_with(
            "rossmann-model", stages=["Production"]
        )
        assert version == "5"

    @patch("src.models.model_registry.get_mlflow_client")
    def test_get_model_version_staging(self, mock_get_client, mock_model_version):
        """Test retrieving Staging model version."""
        # Arrange
        mock_client = Mock()
        mock_model_version.current_stage = "Staging"
        mock_model_version.version = "7"
        mock_client.get_latest_versions.return_value = [mock_model_version]
        mock_get_client.return_value = mock_client

        # Act
        version = model_registry.get_model_version("rossmann-model", stage="Staging")

        # Assert
        mock_client.get_latest_versions.assert_called_once_with(
            "rossmann-model", stages=["Staging"]
        )
        assert version == "7"

    @patch("src.models.model_registry.get_mlflow_client")
    def test_get_model_version_no_model_found(self, mock_get_client):
        """Test get_model_version returns None when no model in stage."""
        # Arrange
        mock_client = Mock()
        mock_client.get_latest_versions.return_value = []  # No models
        mock_get_client.return_value = mock_client

        # Act
        version = model_registry.get_model_version("rossmann-model", stage="Production")

        # Assert
        assert version is None


class TestLoadModel:
    """Tests for load_model()."""

    @patch("src.models.model_registry.mlflow.pyfunc.load_model")
    def test_load_model_by_production_stage(self, mock_load_model):
        """Test loading model by Production stage."""
        # Arrange
        mock_model = Mock()
        mock_load_model.return_value = mock_model

        # Act
        model = model_registry.load_model("rossmann-model", stage="Production")

        # Assert
        mock_load_model.assert_called_once_with("models:/rossmann-model/Production")
        assert model == mock_model

    @patch("src.models.model_registry.mlflow.pyfunc.load_model")
    def test_load_model_by_staging_stage(self, mock_load_model):
        """Test loading model by Staging stage."""
        # Arrange
        mock_model = Mock()
        mock_load_model.return_value = mock_model

        # Act
        model = model_registry.load_model("rossmann-model", stage="Staging")

        # Assert
        mock_load_model.assert_called_once_with("models:/rossmann-model/Staging")
        assert model == mock_model

    @patch("src.models.model_registry.mlflow.pyfunc.load_model")
    def test_load_model_by_version_number(self, mock_load_model):
        """Test loading model by specific version number."""
        # Arrange
        mock_model = Mock()
        mock_load_model.return_value = mock_model

        # Act
        model = model_registry.load_model("rossmann-model", stage="3")

        # Assert
        mock_load_model.assert_called_once_with("models:/rossmann-model/3")
        assert model == mock_model

    @patch("src.models.model_registry.mlflow.pyfunc.load_model")
    def test_load_model_by_archived_stage(self, mock_load_model):
        """Test loading model by Archived stage."""
        # Arrange
        mock_model = Mock()
        mock_load_model.return_value = mock_model

        # Act
        model = model_registry.load_model("rossmann-model", stage="Archived")

        # Assert
        mock_load_model.assert_called_once_with("models:/rossmann-model/Archived")
        assert model == mock_model


class TestGetModelInfo:
    """Tests for get_model_info()."""

    @patch("src.models.model_registry.get_mlflow_client")
    def test_get_model_info_specific_version(self, mock_get_client, mock_model_version):
        """Test getting info for a specific model version."""
        # Arrange
        mock_client = Mock()
        mock_client.get_model_version.return_value = mock_model_version
        mock_get_client.return_value = mock_client

        # Act
        info = model_registry.get_model_info("rossmann-model", version="1")

        # Assert
        mock_client.get_model_version.assert_called_once_with(name="rossmann-model", version="1")
        assert info["name"] == "rossmann-model"
        assert info["version"] == "1"
        assert info["current_stage"] == "Production"
        assert info["run_id"] == "test-run-id-123"
        assert info["status"] == "READY"

    @patch("src.models.model_registry.get_mlflow_client")
    def test_get_model_info_all_versions(
        self, mock_get_client, mock_registered_model, mock_model_version
    ):
        """Test getting info for all versions of a model."""
        # Arrange
        mock_client = Mock()
        mock_client.get_registered_model.return_value = mock_registered_model

        # Create multiple versions
        mv1 = Mock()
        mv1.version = "1"
        mv1.current_stage = "Archived"
        mv1.run_id = "run-1"
        mv1.status = "READY"

        mv2 = Mock()
        mv2.version = "2"
        mv2.current_stage = "Production"
        mv2.run_id = "run-2"
        mv2.status = "READY"

        mock_client.search_model_versions.return_value = [mv1, mv2]
        mock_get_client.return_value = mock_client

        # Act
        info = model_registry.get_model_info("rossmann-model")

        # Assert
        mock_client.get_registered_model.assert_called_once_with("rossmann-model")
        mock_client.search_model_versions.assert_called_once_with("name='rossmann-model'")

        assert info["name"] == "rossmann-model"
        assert len(info["versions"]) == 2
        assert info["versions"][0]["version"] == "1"
        assert info["versions"][0]["stage"] == "Archived"
        assert info["versions"][1]["version"] == "2"
        assert info["versions"][1]["stage"] == "Production"


class TestListRegisteredModels:
    """Tests for list_registered_models()."""

    @patch("src.models.model_registry.get_mlflow_client")
    def test_list_registered_models(self, mock_get_client):
        """Test listing all registered models."""
        # Arrange
        mock_client = Mock()

        rm1 = Mock()
        rm1.name = "rossmann-model"

        rm2 = Mock()
        rm2.name = "rossmann-baseline"

        rm3 = Mock()
        rm3.name = "rossmann-experiment"

        mock_client.search_registered_models.return_value = [rm1, rm2, rm3]
        mock_get_client.return_value = mock_client

        # Act
        models = model_registry.list_registered_models()

        # Assert
        mock_client.search_registered_models.assert_called_once()
        assert models == ["rossmann-model", "rossmann-baseline", "rossmann-experiment"]
        assert len(models) == 3

    @patch("src.models.model_registry.get_mlflow_client")
    def test_list_registered_models_empty(self, mock_get_client):
        """Test listing when no models are registered."""
        # Arrange
        mock_client = Mock()
        mock_client.search_registered_models.return_value = []
        mock_get_client.return_value = mock_client

        # Act
        models = model_registry.list_registered_models()

        # Assert
        assert models == []
        assert len(models) == 0
