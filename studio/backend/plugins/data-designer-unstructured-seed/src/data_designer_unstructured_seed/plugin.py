from data_designer.plugins.plugin import Plugin, PluginType

unstructured_seed_plugin = Plugin(
    impl_qualified_name="data_designer_unstructured_seed.impl.UnstructuredSeedReader",
    config_qualified_name="data_designer_unstructured_seed.config.UnstructuredSeedSource",
    plugin_type=PluginType.SEED_READER,
)
