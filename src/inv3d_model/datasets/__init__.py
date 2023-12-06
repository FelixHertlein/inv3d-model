from .dataset_factory import DatasetFactory
from .doc3d import Doc3DDataset, Doc3DRealDataset, Doc3DTestDataset
from .empty import EmptyDataset
from .inv3d import Inv3DDataset, Inv3DRealDataset, Inv3DTestDataset

# training datasets
DatasetFactory.register_dataset("empty", EmptyDataset)
DatasetFactory.register_dataset("doc3d", Doc3DDataset)
DatasetFactory.register_dataset("inv3d", Inv3DDataset)
DatasetFactory.register_dataset("inv3d_tplwhite", Inv3DDataset, template="white")
DatasetFactory.register_dataset("inv3d_tplstruct", Inv3DDataset, template="struct")
DatasetFactory.register_dataset("inv3d_tpltext", Inv3DDataset, template="text")

# evaluation datasets
DatasetFactory.register_dataset("doc3d_test", Doc3DTestDataset)
DatasetFactory.register_dataset("doc3d_real", Doc3DRealDataset)

DatasetFactory.register_dataset("inv3d_test", Inv3DTestDataset)
DatasetFactory.register_dataset(
    "inv3d_test_tplwhite", Inv3DTestDataset, source="inv3d_test", template="white"
)
DatasetFactory.register_dataset(
    "inv3d_test_tplstruct", Inv3DTestDataset, source="inv3d_test", template="struct"
)
DatasetFactory.register_dataset(
    "inv3d_test_tpltext", Inv3DTestDataset, source="inv3d_test", template="text"
)
DatasetFactory.register_dataset(
    "inv3d_test_tplrandom", Inv3DTestDataset, source="inv3d_test", template="random"
)

DatasetFactory.register_dataset("inv3d_real", Inv3DRealDataset)
DatasetFactory.register_dataset(
    "inv3d_real_tplwhite", Inv3DRealDataset, source="inv3d_real", template="white"
)
DatasetFactory.register_dataset(
    "inv3d_real_tplstruct", Inv3DRealDataset, source="inv3d_real", template="struct"
)
DatasetFactory.register_dataset(
    "inv3d_real_tpltext", Inv3DRealDataset, source="inv3d_real", template="text"
)
DatasetFactory.register_dataset(
    "inv3d_real_tplrandom", Inv3DRealDataset, source="inv3d_real", template="random"
)
