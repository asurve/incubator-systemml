/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
package org.apache.sysml.runtime.instructions.spark.functions;

import java.util.Iterator;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.spark.api.java.function.FlatMapFunction;

import scala.Tuple2;

import org.apache.sysml.runtime.matrix.data.FrameBinaryBlockToTextCellConverter;
import org.apache.sysml.runtime.matrix.data.FrameBlock;
import org.apache.sysml.runtime.matrix.data.Pair;

public class ConvertFrameBlockToIJVLines implements FlatMapFunction<Tuple2<LongWritable,FrameBlock>, String> {

	private static final long serialVersionUID = 1803516615963340115L;

	int brlen; int bclen;
	public ConvertFrameBlockToIJVLines(int brlen, int bclen) {
		this.brlen = brlen;
		this.bclen = bclen;
	}
	
	@Override
	public Iterable<String> call(Tuple2<LongWritable, FrameBlock> kv) throws Exception {
		final FrameBinaryBlockToTextCellConverter converter = new FrameBinaryBlockToTextCellConverter();
		converter.setBlockSize(brlen, bclen);
		converter.convert(kv._1, kv._2);
		
		return new Iterable<String>() {
			@Override
			public Iterator<String> iterator() {
				return new Iterator<String>() {
					
					@Override
					public void remove() {}
					
					@Override
					public String next() {
						Pair <NullWritable, Text> nextText = converter.next();
						if (nextText != null)
							return nextText.getValue().toString();
						else
							return null;
					}
					
					@Override
					public boolean hasNext() {
						return converter.hasNext();
					}
				};
			}
		};
	}

}