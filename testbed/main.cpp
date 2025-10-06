#include <cstdint>
#include <climits>
#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>
#include <numbers>

#include <veekay/veekay.hpp>

#include <imgui.h>
#include <vulkan/vulkan_core.h>

namespace
{

	constexpr float camera_fov = 70.0f;
	constexpr float camera_near_plane = 0.01f;
	constexpr float camera_far_plane = 100.0f;

	struct Matrix
	{
		float m[4][4];
	};

	struct Vector
	{
		float x, y, z;
	};

	struct Vertex
	{
		Vector position;
	};

	struct ShaderConstants
	{
		Matrix projection;
		Matrix transform;
		Vector color;
	};

	struct VulkanBuffer
	{
		VkBuffer buffer;
		VkDeviceMemory memory;
	};

	struct SceneObject
	{
		VulkanBuffer vertex_buffer;
		VulkanBuffer index_buffer;
		uint32_t index_count;

		Vector position;
		Vector color;
		float rotation;
	};

	VkShaderModule vertex_shader_module;
	VkShaderModule fragment_shader_module;
	VkPipelineLayout pipeline_layout;
	VkPipeline pipeline;

	SceneObject cube;
	SceneObject pyramid;
	SceneObject sphere;

	Matrix identity()
	{
		Matrix result{};
		result.m[0][0] = 1.0f;
		result.m[1][1] = 1.0f;
		result.m[2][2] = 1.0f;
		result.m[3][3] = 1.0f;
		return result;
	}

	Matrix projection(float fov, float aspect_ratio, float near, float far)
	{
		Matrix result{};
		const float radians = fov * std::numbers::pi_v<float> / 180.0f;
		const float cot = 1.0f / tanf(radians / 2.0f);
		result.m[0][0] = cot / aspect_ratio;
		result.m[1][1] = cot;
		result.m[2][2] = far / (far - near);
		result.m[2][3] = 1.0f;
		result.m[3][2] = (-near * far) / (far - near);
		return result;
	}

	Matrix translation(Vector vector)
	{
		Matrix result = identity();
		result.m[3][0] = vector.x;
		result.m[3][1] = vector.y;
		result.m[3][2] = vector.z;
		return result;
	}

	Matrix rotation(Vector axis, float angle)
	{
		Matrix result{};
		float length = sqrtf(axis.x * axis.x + axis.y * axis.y + axis.z * axis.z);
		axis.x /= length;
		axis.y /= length;
		axis.z /= length;
		float sina = sinf(angle);
		float cosa = cosf(angle);
		float cosv = 1.0f - cosa;
		result.m[0][0] = (axis.x * axis.x * cosv) + cosa;
		result.m[0][1] = (axis.x * axis.y * cosv) + (axis.z * sina);
		result.m[0][2] = (axis.x * axis.z * cosv) - (axis.y * sina);
		result.m[1][0] = (axis.y * axis.x * cosv) - (axis.z * sina);
		result.m[1][1] = (axis.y * axis.y * cosv) + cosa;
		result.m[1][2] = (axis.y * axis.z * cosv) + (axis.x * sina);
		result.m[2][0] = (axis.z * axis.x * cosv) + (axis.y * sina);
		result.m[2][1] = (axis.z * axis.y * cosv) - (axis.x * sina);
		result.m[2][2] = (axis.z * axis.z * cosv) + cosa;
		result.m[3][3] = 1.0f;
		return result;
	}

	Matrix multiply(const Matrix &a, const Matrix &b)
	{
		Matrix result{};
		for (int j = 0; j < 4; j++)
		{
			for (int i = 0; i < 4; i++)
			{
				for (int k = 0; k < 4; k++)
				{
					result.m[j][i] += a.m[j][k] * b.m[k][i];
				}
			}
		}
		return result;
	}

	VkShaderModule loadShaderModule(const char *path)
	{
		std::ifstream file(path, std::ios::binary | std::ios::ate);
		if (!file.is_open())
		{
			return nullptr;
		}
		size_t size = file.tellg();
		std::vector<uint32_t> buffer(size / sizeof(uint32_t));
		file.seekg(0);
		file.read(reinterpret_cast<char *>(buffer.data()), size);
		file.close();
		VkShaderModuleCreateInfo info{
			.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
			.codeSize = size,
			.pCode = buffer.data(),
		};
		VkShaderModule result;
		vkCreateShaderModule(veekay::app.vk_device, &info, nullptr, &result);
		return result;
	}

	VulkanBuffer createBuffer(size_t size, void *data, VkBufferUsageFlags usage)
	{
		VkDevice &device = veekay::app.vk_device;
		VkPhysicalDevice &physical_device = veekay::app.vk_physical_device;
		VulkanBuffer result{};
		VkBufferCreateInfo bufferInfo{
			.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
			.size = size,
			.usage = usage,
			.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		};
		vkCreateBuffer(device, &bufferInfo, nullptr, &result.buffer);
		VkMemoryRequirements memRequirements;
		vkGetBufferMemoryRequirements(device, result.buffer, &memRequirements);
		VkPhysicalDeviceMemoryProperties memProperties;
		vkGetPhysicalDeviceMemoryProperties(physical_device, &memProperties);
		const VkMemoryPropertyFlags flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
		uint32_t memoryTypeIndex = UINT_MAX;
		for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++)
		{
			if ((memRequirements.memoryTypeBits & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & flags) == flags)
			{
				memoryTypeIndex = i;
				break;
			}
		}
		VkMemoryAllocateInfo allocInfo{
			.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
			.allocationSize = memRequirements.size,
			.memoryTypeIndex = memoryTypeIndex,
		};
		vkAllocateMemory(device, &allocInfo, nullptr, &result.memory);
		vkBindBufferMemory(device, result.buffer, result.memory, 0);
		void *mapped_data;
		vkMapMemory(device, result.memory, 0, size, 0, &mapped_data);
		memcpy(mapped_data, data, size);
		vkUnmapMemory(device, result.memory);
		return result;
	}

	void destroyBuffer(const VulkanBuffer &buffer)
	{
		if (buffer.buffer != VK_NULL_HANDLE)
		{
			vkDestroyBuffer(veekay::app.vk_device, buffer.buffer, nullptr);
		}
		if (buffer.memory != VK_NULL_HANDLE)
		{
			vkFreeMemory(veekay::app.vk_device, buffer.memory, nullptr);
		}
	}

	void initialize()
	{
		VkDevice &device = veekay::app.vk_device;

		{ // Build graphics pipeline
			vertex_shader_module = loadShaderModule("./shaders/shader.vert.spv");
			fragment_shader_module = loadShaderModule("./shaders/shader.frag.spv");

			VkPipelineShaderStageCreateInfo stage_infos[2] = {};
			stage_infos[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
			stage_infos[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
			stage_infos[0].module = vertex_shader_module;
			stage_infos[0].pName = "main";
			stage_infos[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
			stage_infos[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
			stage_infos[1].module = fragment_shader_module;
			stage_infos[1].pName = "main";

			VkVertexInputBindingDescription buffer_binding{
				.binding = 0,
				.stride = sizeof(Vertex),
				.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
			};
			VkVertexInputAttributeDescription attributes[]{
				{0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position)},
			};
			VkPipelineVertexInputStateCreateInfo input_state_info{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
				.vertexBindingDescriptionCount = 1,
				.pVertexBindingDescriptions = &buffer_binding,
				.vertexAttributeDescriptionCount = 1,
				.pVertexAttributeDescriptions = attributes,
			};
			VkPipelineInputAssemblyStateCreateInfo assembly_state_info{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
				.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
			};
			VkPipelineRasterizationStateCreateInfo raster_info{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
				.polygonMode = VK_POLYGON_MODE_FILL,
				.cullMode = VK_CULL_MODE_BACK_BIT,
				.frontFace = VK_FRONT_FACE_CLOCKWISE,
				.lineWidth = 1.0f,
			};
			VkPipelineMultisampleStateCreateInfo sample_info{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
				.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
			};
			VkViewport viewport{
				.width = (float)veekay::app.window_width,
				.height = (float)veekay::app.window_height,
				.maxDepth = 1.0f,
			};
			VkRect2D scissor{.extent = {veekay::app.window_width, veekay::app.window_height}};
			VkPipelineViewportStateCreateInfo viewport_info{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
				.viewportCount = 1,
				.pViewports = &viewport,
				.scissorCount = 1,
				.pScissors = &scissor,
			};
			VkPipelineDepthStencilStateCreateInfo depth_info{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
				.depthTestEnable = VK_TRUE,
				.depthWriteEnable = VK_TRUE,
				.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
			};
			VkPipelineColorBlendAttachmentState attachment_info{
				.colorWriteMask = 0xF,
			};
			VkPipelineColorBlendStateCreateInfo blend_info{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
				.attachmentCount = 1,
				.pAttachments = &attachment_info,
			};
			VkPushConstantRange push_constants{
				.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
				.size = sizeof(ShaderConstants),
			};
			VkPipelineLayoutCreateInfo layout_info{
				.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
				.pushConstantRangeCount = 1,
				.pPushConstantRanges = &push_constants,
			};
			vkCreatePipelineLayout(device, &layout_info, nullptr, &pipeline_layout);
			VkGraphicsPipelineCreateInfo info{
				.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
				.stageCount = 2,
				.pStages = stage_infos,
				.pVertexInputState = &input_state_info,
				.pInputAssemblyState = &assembly_state_info,
				.pViewportState = &viewport_info,
				.pRasterizationState = &raster_info,
				.pMultisampleState = &sample_info,
				.pDepthStencilState = &depth_info,
				.pColorBlendState = &blend_info,
				.layout = pipeline_layout,
				.renderPass = veekay::app.vk_render_pass,
			};
			vkCreateGraphicsPipelines(device, nullptr, 1, &info, nullptr, &pipeline);
		}

		// Create Cube
		{
			Vertex vertices[] = {
				{{-0.5f, -0.5f, 0.5f}}, {{0.5f, -0.5f, 0.5f}}, {{0.5f, 0.5f, 0.5f}}, {{-0.5f, 0.5f, 0.5f}}, {{-0.5f, -0.5f, -0.5f}}, {{0.5f, -0.5f, -0.5f}}, {{0.5f, 0.5f, -0.5f}}, {{-0.5f, 0.5f, -0.5f}}};
			uint32_t indices[] = {
				0, 1, 2, 2, 3, 0, 1, 5, 6, 6, 2, 1, 7, 6, 5, 5, 4, 7,
				4, 0, 3, 3, 7, 4, 4, 5, 1, 1, 0, 4, 3, 2, 6, 6, 7, 3};
			cube.vertex_buffer = createBuffer(sizeof(vertices), vertices, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
			cube.index_buffer = createBuffer(sizeof(indices), indices, VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
			cube.index_count = sizeof(indices) / sizeof(uint32_t);
			cube.position = {-2.0f, 0.0f, 7.0f};
			cube.color = {1.0f, 0.0f, 0.0f};
		}

		// Create Pyramid
		{
			Vertex vertices[] = {
				{{0.0f, 0.5f, 0.0f}}, {{-0.5f, -0.5f, 0.5f}}, {{0.5f, -0.5f, 0.5f}}, {{0.5f, -0.5f, -0.5f}}, {{-0.5f, -0.5f, -0.5f}}};
			uint32_t indices[] = {
				0, 1, 2, 0, 2, 3, 0, 3, 4, 0, 4, 1, 1, 4, 3, 3, 2, 1};
			pyramid.vertex_buffer = createBuffer(sizeof(vertices), vertices, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
			pyramid.index_buffer = createBuffer(sizeof(indices), indices, VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
			pyramid.index_count = sizeof(indices) / sizeof(uint32_t);
			pyramid.position = {0.0f, 0.0f, 7.0f};
			pyramid.color = {0.0f, 1.0f, 0.0f};
		}

		// Create Sphere
		{
			std::vector<Vertex> vertices;
			std::vector<uint32_t> indices;
			const int sectors = 36;
			const int stacks = 18;
			const float radius = 0.5f;
			for (int i = 0; i <= stacks; ++i)
			{
				float stackAngle = std::numbers::pi_v<float> / 2 - i * std::numbers::pi_v<float> / stacks;
				float xy = radius * cosf(stackAngle);
				float z = radius * sinf(stackAngle);
				for (int j = 0; j <= sectors; ++j)
				{
					float sectorAngle = j * 2 * std::numbers::pi_v<float> / sectors;
					float x = xy * cosf(sectorAngle);
					float y = xy * sinf(sectorAngle);
					vertices.push_back({{x, y, z}});
				}
			}
			for (int i = 0; i < stacks; ++i)
			{
				for (int j = 0; j < sectors; ++j)
				{
					int first = (i * (sectors + 1)) + j;
					int second = first + sectors + 1;
					indices.push_back(first);
					indices.push_back(second);
					indices.push_back(first + 1);
					indices.push_back(second);
					indices.push_back(second + 1);
					indices.push_back(first + 1);
				}
			}
			sphere.vertex_buffer = createBuffer(vertices.size() * sizeof(Vertex), vertices.data(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
			sphere.index_buffer = createBuffer(indices.size() * sizeof(uint32_t), indices.data(), VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
			sphere.index_count = indices.size();
			sphere.position = {2.0f, 0.0f, 7.0f};
			sphere.color = {0.0f, 0.0f, 1.0f};
		}
	}

	void shutdown()
	{
		VkDevice &device = veekay::app.vk_device;
		destroyBuffer(cube.index_buffer);
		destroyBuffer(cube.vertex_buffer);
		destroyBuffer(pyramid.index_buffer);
		destroyBuffer(pyramid.vertex_buffer);
		destroyBuffer(sphere.index_buffer);
		destroyBuffer(sphere.vertex_buffer);
		vkDestroyPipeline(device, pipeline, nullptr);
		vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
		vkDestroyShaderModule(device, fragment_shader_module, nullptr);
		vkDestroyShaderModule(device, vertex_shader_module, nullptr);
	}

	void update(double time)
	{
		cube.rotation = fmodf(float(time) * 0.5f, 2.0f * std::numbers::pi_v<float>);
		pyramid.rotation = fmodf(float(time) * 0.5f, 2.0f * std::numbers::pi_v<float>);
		sphere.rotation = fmodf(float(time) * 0.5f, 2.0f * std::numbers::pi_v<float>);
	}

	void render(VkCommandBuffer cmd, VkFramebuffer framebuffer)
	{
		vkResetCommandBuffer(cmd, 0);
		VkCommandBufferBeginInfo beginInfo{
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
			.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
		};
		vkBeginCommandBuffer(cmd, &beginInfo);

		VkClearValue clear_values[2];
		clear_values[0].color = {{0.1f, 0.1f, 0.1f, 1.0f}};
		clear_values[1].depthStencil = {1.0f, 0};
		VkRenderPassBeginInfo renderPassInfo{
			.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
			.renderPass = veekay::app.vk_render_pass,
			.framebuffer = framebuffer,
			.renderArea = {.extent = {veekay::app.window_width, veekay::app.window_height}},
			.clearValueCount = 2,
			.pClearValues = clear_values,
		};
		vkCmdBeginRenderPass(cmd, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

		auto draw_object = [&](const SceneObject &obj)
		{
			VkDeviceSize offset = 0;
			vkCmdBindVertexBuffers(cmd, 0, 1, &obj.vertex_buffer.buffer, &offset);
			vkCmdBindIndexBuffer(cmd, obj.index_buffer.buffer, 0, VK_INDEX_TYPE_UINT32);
			ShaderConstants constants{
				.projection = projection(
					camera_fov,
					(float)veekay::app.window_width / (float)veekay::app.window_height,
					camera_near_plane, camera_far_plane),
				.transform = multiply(rotation({0.0f, 1.0f, 0.0f}, obj.rotation), translation(obj.position)),
				.color = obj.color,
			};
			vkCmdPushConstants(cmd, pipeline_layout,
							   VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
							   0, sizeof(ShaderConstants), &constants);
			vkCmdDrawIndexed(cmd, obj.index_count, 1, 0, 0, 0);
		};

		draw_object(cube);
		draw_object(pyramid);
		draw_object(sphere);

		vkCmdEndRenderPass(cmd);
		vkEndCommandBuffer(cmd);
	}

} // namespace

int main()
{
	return veekay::run({
		.init = initialize,
		.shutdown = shutdown,
		.update = update,
		.render = render,
	});
}