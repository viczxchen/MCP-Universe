import unittest
from mcpuniverse.mcp.manager import MCPManager


class TestBilibiliVideoTool(unittest.IsolatedAsyncioTestCase):

    async def test_server_tools(self):
        """Test that all expected tools are available."""
        manager = MCPManager()
        client = await manager.build_client(server_name="bilibili-video-tool")
        tools = await client.list_tools()
        tool_names = [tool.name for tool in tools]
        
        # Verify all four tools are present
        self.assertIn("search_video", tool_names)
        self.assertIn("download_video", tool_names)
        self.assertIn("get_comments", tool_names)
        self.assertIn("get_subtitles", tool_names)
        self.assertEqual(len(tools), 4)
        
        await client.cleanup()

    async def test_search_video_tool(self):
        """Test search_video tool structure."""
        manager = MCPManager()
        client = await manager.build_client(server_name="bilibili-video-tool")
        tools = await client.list_tools()
        
        search_tool = next((tool for tool in tools if tool.name == "search_video"), None)
        self.assertIsNotNone(search_tool)
        self.assertIn("query", str(search_tool.inputSchema))
        
        await client.cleanup()

    async def test_download_video_tool(self):
        """Test download_video tool structure."""
        manager = MCPManager()
        client = await manager.build_client(server_name="bilibili-video-tool")
        tools = await client.list_tools()
        
        download_tool = next((tool for tool in tools if tool.name == "download_video"), None)
        self.assertIsNotNone(download_tool)
        self.assertIn("video_id", str(download_tool.inputSchema))
        
        await client.cleanup()

    async def test_get_comments_tool(self):
        """Test get_comments tool structure."""
        manager = MCPManager()
        client = await manager.build_client(server_name="bilibili-video-tool")
        tools = await client.list_tools()
        
        comments_tool = next((tool for tool in tools if tool.name == "get_comments"), None)
        self.assertIsNotNone(comments_tool)
        self.assertIn("video_id", str(comments_tool.inputSchema))
        
        await client.cleanup()

    async def test_get_subtitles_tool(self):
        """Test get_subtitles tool structure."""
        manager = MCPManager()
        client = await manager.build_client(server_name="bilibili-video-tool")
        tools = await client.list_tools()
        
        subtitles_tool = next((tool for tool in tools if tool.name == "get_subtitles"), None)
        self.assertIsNotNone(subtitles_tool)
        self.assertIn("video_id", str(subtitles_tool.inputSchema))
        
        await client.cleanup()


if __name__ == "__main__":
    unittest.main()

